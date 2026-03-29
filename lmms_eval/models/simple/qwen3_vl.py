import base64
import os
import re
import time
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

def _extract_qid_videoid(doc: dict) -> Tuple[str, str]:
    qid = (
        doc.get("question_id")
        or doc.get("questionId")
        or doc.get("questionID")
        or doc.get("qid")
        or doc.get("id")
        or doc.get("doc_id")
        or doc.get("docId")
        or "na"
    )
    vid = (
        doc.get("video_id")
        or doc.get("videoId")
        or doc.get("videoID")
        or doc.get("video")
        or doc.get("video_name")
        or doc.get("video_uid")
        or "na"
    )
    return str(qid), str(vid)


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        # max_pixels: int = 1605632,
        max_pixels: int = 12845056,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        self.total_cuda_time = 0
        self.max_mem = 0

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # check whether its an MoE model
        match = re.search(r"A\d+B", pretrained)
        model_fn = Qwen3VLMoeForConditionalGeneration if match else Qwen3VLForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()

        # =======================================================
        #  Start of Compression Model Wrapper Logic
        # =======================================================
        def _sync_dense_patches_to_moe() -> None:
            """
            Many compressors monkey-patch `transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel`.
            For A*B checkpoints (e.g. Qwen3-VL-30B-A3B-Instruct), HF uses MoE classes
            under a different module (`transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe`).
            To make all compressors work for both dense and MoE variants, mirror patched
            methods from the dense class onto the MoE model class(es) when present.
            """
            try:
                from transformers.models.qwen3_vl import modeling_qwen3_vl as dense_modeling
            except Exception:
                return

            dense_cls = getattr(dense_modeling, "Qwen3VLModel", None)
            if dense_cls is None:
                return

            patched_forward = getattr(dense_cls, "forward", None)
            patched_get_video_features = getattr(dense_cls, "get_video_features", None)
            patched_compute_text_instruction_embedding = getattr(dense_cls, "_compute_text_instruction_embedding", None)

            # Prefer patching the dedicated MoE module if present (newer HF versions).
            try:
                import importlib

                moe_modeling = importlib.import_module("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")  # type: ignore
            except Exception:
                moe_modeling = None

            patched_any = False
            patched_classes = []
            debug = os.environ.get("QWEN3_PATCHSYNC_DEBUG", "0") == "1"
            if moe_modeling is not None:
                # IMPORTANT:
                # patch only the MoE VL backbone class. Do NOT patch any other MoE classes
                # (e.g. Qwen3VLMoeTextModel / Qwen3VLMoeForConditionalGeneration), otherwise
                # methods expecting a backbone `self` will bind to the wrong module.
                moe_cls = getattr(moe_modeling, "Qwen3VLMoeModel", None)
                if moe_cls is not None and hasattr(moe_cls, "forward"):
                    if patched_forward is not None:
                        setattr(moe_cls, "forward", patched_forward)
                    if patched_get_video_features is not None and hasattr(moe_cls, "get_video_features"):
                        setattr(moe_cls, "get_video_features", patched_get_video_features)
                    if (
                        patched_compute_text_instruction_embedding is not None
                        and hasattr(moe_cls, "_compute_text_instruction_embedding")
                    ):
                        setattr(moe_cls, "_compute_text_instruction_embedding", patched_compute_text_instruction_embedding)
                    patched_any = True
                    patched_classes.append("Qwen3VLMoeModel")

                if debug and patched_any:
                    try:
                        fn = getattr(getattr(moe_modeling, "Qwen3VLMoeModel", None), "get_video_features", None)
                        fn_mod = getattr(fn, "__module__", None) if fn is not None else None
                    except Exception:
                        fn_mod = None
                    print(f"[PatchSync][Debug] Patched MoE classes: {sorted(set(patched_classes))}")
                    if fn_mod:
                        print(f"[PatchSync][Debug] MoE get_video_features now from: {fn_mod}")

            # Fallback: some versions may still expose MoE model class in the dense module.
            if not patched_any:
                moe_cls = getattr(dense_modeling, "Qwen3VLMoeModel", None)
                if moe_cls is not None and hasattr(moe_cls, "get_video_features"):
                    if patched_forward is not None and hasattr(moe_cls, "forward"):
                        setattr(moe_cls, "forward", patched_forward)
                    if patched_get_video_features is not None:
                        setattr(moe_cls, "get_video_features", patched_get_video_features)
                    if patched_compute_text_instruction_embedding is not None and hasattr(moe_cls, "_compute_text_instruction_embedding"):
                        setattr(moe_cls, "_compute_text_instruction_embedding", patched_compute_text_instruction_embedding)
                    patched_any = True

            if patched_any:
                print("[PatchSync] Mirrored dense Qwen3VLModel patches onto MoE Qwen3-VL model classes.")
            elif debug:
                print("[PatchSync][Debug] No MoE model classes patched (module not found or pattern mismatch).")

        import importlib
        model_wrapper = (os.environ.get("method") or os.environ.get("METHOD") or "v_cast").strip()

        if model_wrapper != "v_cast":
            raise ValueError(f"Unsupported method '{model_wrapper}'. This repository only keeps 'v_cast'.")

        print("Loading Sparse Mode: v_cast from 'compressor' package")
        try:
            wrapper_module = importlib.import_module("compressor.v_cast")
            wrapper_class = getattr(wrapper_module, "v_cast")
            self._model = wrapper_class(self._model)
            if match:
                _sync_dense_patches_to_moe()
        except ImportError:
            eval_logger.error("Failed to import module 'v_cast'. Make sure it is installed or in the python path.")
            raise
        except AttributeError:
            eval_logger.error("Module 'v_cast' does not have a callable named 'v_cast'.")
            raise
        # =======================================================
        # End of Wrapper Logic
        # =======================================================

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        print_per_sample_timing = os.environ.get("QWEN3VL_PRINT_TIMING_PER_SAMPLE", "0") == "1"
        if not hasattr(self, "timing_total_prefill_ms"):
            self.timing_total_prefill_ms = 0.0
        if not hasattr(self, "timing_total_decode_ms"):
            self.timing_total_decode_ms = 0.0
        if not hasattr(self, "timing_llm_prefill_ms"):
            self.timing_llm_prefill_ms = 0.0

        def _safe_nframes_for_video(total_frames: Optional[int], requested: int) -> int:
            """
            qwen_vl_utils enforces FRAME_FACTOR (typically 2) and requires nframes <= total_frames.
            MVBench contains short/odd-length videos (e.g., 31 frames) where requesting 32 triggers:
              ValueError: nframes should in interval [2, 31], but got 32.
            We conservatively cap to an even <= total_frames when possible.
            """
            if total_frames is None:
                return int(requested)
            total_frames = int(total_frames)
            requested = int(requested)
            if total_frames <= 0:
                return requested
            # Keep within bounds.
            capped = min(requested, total_frames)
            # Prefer an even number when possible (FRAME_FACTOR=2).
            if capped > 2 and (capped % 2 == 1):
                capped = capped - 1
            # Respect lower bound when possible.
            if total_frames >= 2:
                capped = max(2, min(capped, total_frames))
            return capped

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            # Expose current sample ids to compressor logs (batch_size=1 in our eval scripts).
            try:
                sample_doc = self.task_dict[task][split][doc_id[0]]
                qid, vid = _extract_qid_videoid(sample_doc)
                os.environ["QWEN3VL_CUR_QUESTION_ID"] = qid
                os.environ["QWEN3VL_CUR_VIDEO_ID"] = vid
            except Exception:
                os.environ["QWEN3VL_CUR_QUESTION_ID"] = str(doc_id[0]) if len(doc_id) > 0 else "na"
                os.environ["QWEN3VL_CUR_VIDEO_ID"] = "na"

            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            total_frames = len(vr)
                            video_item = {
                                "type": "video",
                                "video": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                            if self.fps is not None:
                                video_item["fps"] = self.fps
                                video_item["max_frames"] = _safe_nframes_for_video(total_frames, self.max_num_frames)
                            else:
                                video_item["nframes"] = _safe_nframes_for_video(total_frames, self.max_num_frames)
                            processed_visuals.append(video_item)
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            processed_visuals.append({"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            # TODO: refactor code to allow return_video_kwargs and return_video_metadata
            try:
                image_inputs, video_inputs = process_vision_info(
                    batched_messages, return_video_kwargs=False, image_patch_size=16, return_video_metadata=False
                )
            except ValueError as exc:
                # Defensive fallback for qwen_vl_utils nframes bounds issue on short/odd-length videos.
                msg = str(exc)
                match = re.search(r"nframes should in interval \\[2, (\\d+)\\], but got (\\d+)", msg)
                if match:
                    total_frames = int(match.group(1))
                    safe_total = total_frames - 1 if total_frames > 2 else total_frames
                    # Patch all video items in-place and retry once.
                    for m in batched_messages:
                        for item in m:
                            if item.get("type") == "video":
                                if self.fps is not None:
                                    item["max_frames"] = min(int(item.get("max_frames", self.max_num_frames)), safe_total)
                                else:
                                    item["nframes"] = min(int(item.get("nframes", self.max_num_frames)), safe_total)
                    image_inputs, video_inputs = process_vision_info(
                        batched_messages, return_video_kwargs=False, image_patch_size=16, return_video_metadata=False
                    )
                else:
                    raise
            # -----------------------
            if self.batch_size > 1:
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, do_resize=False, padding=True, padding_side="left", return_tensors="pt")
            else:
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, do_resize=False, return_tensors="pt")
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gen_start_event = torch.cuda.Event(enable_timing=True)
                gen_end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                gen_start_event.record()

            prefill_ms = 0.0
            decode_ms = 0.0
            forward_calls = 0
            llm_prefill_ms = 0.0
            orig_forward = self.model.forward

            def _get_llm_submodule(maybe_conditional):
                backbone = getattr(maybe_conditional, "model", None)
                if backbone is None:
                    return None
                for name in ("model", "language_model", "llm"):
                    sub = getattr(backbone, name, None)
                    if sub is not None and callable(getattr(sub, "forward", None)):
                        return sub
                return None

            def _timed_forward(*f_args, **f_kwargs):
                nonlocal prefill_ms, decode_ms, forward_calls, llm_prefill_ms
                is_prefill = forward_calls == 0
                if torch.cuda.is_available():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start.record()
                    llm_sub = _get_llm_submodule(self.model)
                    if is_prefill and llm_sub is not None:
                        orig_llm_forward = llm_sub.forward

                        def _timed_llm_forward(*lm_args, **lm_kwargs):
                            nonlocal llm_prefill_ms
                            s = torch.cuda.Event(enable_timing=True)
                            e = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            s.record()
                            o = orig_llm_forward(*lm_args, **lm_kwargs)
                            e.record()
                            torch.cuda.synchronize()
                            llm_prefill_ms += s.elapsed_time(e)
                            return o

                        try:
                            llm_sub.forward = _timed_llm_forward
                            out = orig_forward(*f_args, **f_kwargs)
                        finally:
                            llm_sub.forward = orig_llm_forward
                    else:
                        out = orig_forward(*f_args, **f_kwargs)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end)
                else:
                    t0 = time.time()
                    out = orig_forward(*f_args, **f_kwargs)
                    elapsed = (time.time() - t0) * 1000.0
                if is_prefill:
                    prefill_ms += elapsed
                else:
                    decode_ms += elapsed
                forward_calls += 1
                return out

            try:
                self.model.forward = _timed_forward
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
            finally:
                self.model.forward = orig_forward

            if torch.cuda.is_available():
                gen_end_event.record()
                torch.cuda.synchronize()
                gen_time = gen_start_event.elapsed_time(gen_end_event) / 1000.0
                gen_max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.total_cuda_time += gen_time
                self.max_mem = max(gen_max_mem, self.max_mem)

            if print_per_sample_timing:
                eval_logger.info(
                    f"[Timing] prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f} total_ms={prefill_ms + decode_ms:.2f}"
                )
            self.timing_total_prefill_ms += prefill_ms
            self.timing_total_decode_ms += decode_ms
            self.timing_llm_prefill_ms += llm_prefill_ms

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Raw Response: {ans}")
                # eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

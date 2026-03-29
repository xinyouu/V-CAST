import os
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLVisionModel,
)


DEFAULT_RETAIN_RATIO = 0.25
DEFAULT_MIN_K = 1
DEFAULT_BUDGET_TEMP = 0.7


def _v_cast_print() -> bool:
    return True


def _tokens_per_frame(grid_thw_row: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int, int]:
    t = int(grid_thw_row[0].item())
    h = int(grid_thw_row[1].item())
    w = int(grid_thw_row[2].item())
    s = max(1, int(spatial_merge_size))
    resize_h = max(1, h // s)
    resize_w = max(1, w // s)
    return t, int(resize_h), int(resize_w), int(resize_h * resize_w)


def _compute_curvature(frame_reps: torch.Tensor) -> torch.Tensor:
    """
    frame_reps: [T, D], normalized
    returns curvature: [T]
    """
    t = int(frame_reps.shape[0])
    if t <= 1:
        return torch.ones((t,), device=frame_reps.device, dtype=torch.float32)
    v_in = frame_reps[1:-1] - frame_reps[:-2]
    v_out = frame_reps[2:] - frame_reps[1:-1]
    curv = 1.0 - F.cosine_similarity(v_in, v_out, dim=-1, eps=1e-6)
    ones = torch.ones((1,), device=frame_reps.device, dtype=curv.dtype)
    return torch.cat([ones, curv, ones], dim=0)


def _allocate_budget_per_frame(
    curvature: torch.Tensor,
    total_budget: int,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    """
    Allocate per-frame budgets summing to total_budget.
    curvature: [T] float
    returns: [T] long
    """
    t = int(curvature.shape[0])
    if t <= 0:
        return torch.zeros((0,), device=curvature.device, dtype=torch.long)

    total_budget = int(max(0, min(int(total_budget), int(t * max_k))))
    min_k = int(max(0, min(int(min_k), int(max_k))))

    if total_budget < t * min_k:
        min_k_eff = 0
    else:
        min_k_eff = min_k

    base = torch.full((t,), int(min_k_eff), device=curvature.device, dtype=torch.long)
    remaining = int(total_budget - int(base.sum().item()))
    if remaining <= 0:
        return base

    weights = curvature.float().clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-6)

    raw = weights * float(remaining)
    extra = torch.floor(raw).to(torch.long)
    max_extra = max_k - min_k_eff
    if max_extra < 0:
        max_extra = 0
    extra = torch.minimum(extra, torch.full_like(extra, int(max_extra)))
    alloc = base + extra

    remaining = int(total_budget - int(alloc.sum().item()))
    if remaining > 0:
        frac = raw - torch.floor(raw)
        frac = frac.masked_fill(alloc >= max_k, -1.0)
        for _ in range(remaining):
            idx = torch.argmax(frac)
            if frac[idx].item() < 0:
                break
            alloc[idx] += 1
            if alloc[idx] >= max_k:
                frac[idx] = -1.0
    elif remaining < 0:
        over = -remaining
        order = torch.argsort(weights, descending=False)
        for idx in order:
            if over <= 0:
                break
            if alloc[idx] > min_k_eff:
                alloc[idx] -= 1
                over -= 1

    return alloc


def _v_cast_compress_frames(
    frames: torch.Tensor,  # [T, HW, D]
    deepstack_frames: List[torch.Tensor],  # list of [T, HW, D_ds]
    retain_ratio: float,
    min_k: int,
    tag: str = "",
) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    t, tokens_per_frame, dim = frames.shape
    if t <= 0 or tokens_per_frame <= 0:
        keep_index = torch.zeros((0,), device=frames.device, dtype=torch.long)
        empty_main = frames.new_empty((0, dim))
        empty_ds = [ds.new_empty((0, ds.shape[-1])) for ds in deepstack_frames]
        return empty_main, empty_ds, keep_index

    if retain_ratio <= 0:
        retain_ratio = DEFAULT_RETAIN_RATIO

    frame_reps = F.normalize(frames.mean(dim=1), dim=-1, eps=1e-6)
    curvature = _compute_curvature(frame_reps)
    weights = torch.softmax(curvature.float() / float(DEFAULT_BUDGET_TEMP), dim=0)

    total_budget = int(round(float(t * tokens_per_frame) * float(retain_ratio)))
    total_budget = max(1, min(int(t * tokens_per_frame), total_budget))
    min_k = int(max(0, min(int(min_k), int(tokens_per_frame))))
    if min_k > 0:
        total_budget = max(total_budget, int(t * min_k))
    k_t = _allocate_budget_per_frame(
        weights,
        total_budget,
        min_k=min_k,
        max_k=tokens_per_frame,
    )

    if _v_cast_print():
        total_keep = int(k_t.sum().item())
        total_tokens = int(t * tokens_per_frame)
        ratio = float(total_keep) / float(total_tokens) if total_tokens > 0 else 0.0
        tag_str = f"{tag} " if tag else ""
        qid = os.environ.get("QWEN3VL_CUR_QUESTION_ID", "na")
        vid = os.environ.get("QWEN3VL_CUR_VIDEO_ID", "na")
        k_list = k_t.tolist()
        ratio_list = [f"{k / float(tokens_per_frame):.4f}" for k in k_list]
        curv_min = float(curvature.min().item()) if curvature.numel() > 0 else 0.0
        curv_mean = float(curvature.mean().item()) if curvature.numel() > 0 else 0.0
        curv_max = float(curvature.max().item()) if curvature.numel() > 0 else 0.0
        w_min = float(weights.min().item()) if weights.numel() > 0 else 0.0
        w_mean = float(weights.mean().item()) if weights.numel() > 0 else 0.0
        w_max = float(weights.max().item()) if weights.numel() > 0 else 0.0
        print(
            f"[V-CAST] {tag_str}frames={t} tokens/frame={tokens_per_frame} retain_ratio={retain_ratio} "
            f"keep={total_keep}/{total_tokens} ({ratio:.4f})"
        )
        print(f"[V-CAST] {tag_str}budget=curvature/softmax@0.7 score=hybrid keep_ends=False")
        print(f"[V-CAST] {tag_str}curvature[min,mean,max]=[{curv_min:.4f},{curv_mean:.4f},{curv_max:.4f}]")
        print(f"[V-CAST] {tag_str}weight[min,mean,max]=[{w_min:.4f},{w_mean:.4f},{w_max:.4f}]")
        print(f"[V-CAST] qid={qid} video_id={vid} {tag_str}frame_keep={k_list}")
        print(f"[V-CAST] qid={qid} video_id={vid} {tag_str}frame_ratio={ratio_list}")

    tokens_out = []
    keep_indices = []
    deepstack_out = [[] for _ in deepstack_frames]

    for frame_idx in range(t):
        k = int(k_t[frame_idx].item())
        if k <= 0:
            continue

        frame_tokens = frames[frame_idx]
        rep = frame_reps[frame_idx]
        sim = F.cosine_similarity(frame_tokens, rep.unsqueeze(0), dim=-1, eps=1e-6)
        outlier = (1.0 - sim).float()
        norm = frame_tokens.float().norm(dim=-1)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
        score = outlier + norm
        topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices
        topk_idx, _ = torch.sort(topk_idx)

        tokens_out.append(frame_tokens.index_select(0, topk_idx))
        keep_indices.append(topk_idx + frame_idx * tokens_per_frame)

        for ds_idx, ds in enumerate(deepstack_frames):
            ds_frame = ds[frame_idx]
            deepstack_out[ds_idx].append(ds_frame.index_select(0, topk_idx))

    if len(tokens_out) == 0:
        keep_index = torch.zeros((0,), device=frames.device, dtype=torch.long)
        empty_main = frames.new_empty((0, dim))
        empty_ds = [ds.new_empty((0, ds.shape[-1])) for ds in deepstack_frames]
        return empty_main, empty_ds, keep_index

    tokens_comp = torch.cat(tokens_out, dim=0)
    keep_index = torch.cat(keep_indices, dim=0).to(torch.long)
    deepstack_comp = [
        torch.cat(chunks, dim=0) if len(chunks) > 0 else ds.new_empty((0, ds.shape[-1]))
        for chunks, ds in zip(deepstack_out, deepstack_frames)
    ]
    return tokens_comp, deepstack_comp, keep_index


def v_cast_compress_qwen3(
    hidden_states: torch.Tensor,
    deepstack_feature_lists: List[torch.Tensor],
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
    retain_ratio: float,
    min_k: int,
) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    total_tokens = int(hidden_states.shape[0])
    if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[1] < 3:
        if _v_cast_print():
            print("[V-CAST] grid_thw invalid; skipping compression.")
        keep_index = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long)
        return hidden_states, [d for d in deepstack_feature_lists], keep_index

    tokens_per_frame_list: list[tuple[int, int]] = []
    expected_total = 0
    for i in range(grid_thw.shape[0]):
        t, _, _, tpf = _tokens_per_frame(grid_thw[i], spatial_merge_size)
        tokens_per_frame_list.append((t, tpf))
        expected_total += int(t * tpf)

    if expected_total != total_tokens:
        if _v_cast_print():
            print(
                f"[V-CAST] token count mismatch; expected {expected_total} from grid_thw, got {total_tokens}. "
                "Skipping compression."
            )
        keep_index = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long)
        return hidden_states, [d for d in deepstack_feature_lists], keep_index

    out_tokens = []
    out_keep = []
    out_deepstack = [[] for _ in deepstack_feature_lists]

    offset = 0
    for vid_idx, (t, tpf) in enumerate(tokens_per_frame_list):
        num = int(t * tpf)
        if num <= 0:
            continue

        chunk = hidden_states[offset : offset + num]
        if t <= 1 or tpf <= 0:
            if _v_cast_print():
                print(f"[V-CAST] video[{vid_idx}] T={t} tpf={tpf}; skipping compression.")
            out_tokens.append(chunk)
            out_keep.append(torch.arange(offset, offset + num, device=hidden_states.device, dtype=torch.long))
            for ds_idx, ds in enumerate(deepstack_feature_lists):
                out_deepstack[ds_idx].append(ds[offset : offset + num])
            offset += num
            continue

        frames = chunk.view(t, tpf, -1)
        ds_frames = [ds[offset : offset + num].view(t, tpf, -1) for ds in deepstack_feature_lists]
        tokens_comp, deepstack_comp, keep_index = _v_cast_compress_frames(
            frames,
            ds_frames,
            retain_ratio=retain_ratio,
            min_k=min_k,
            tag=f"video[{vid_idx}]",
        )
        out_tokens.append(tokens_comp)
        out_keep.append(keep_index + offset)
        for ds_idx, ds_comp in enumerate(deepstack_comp):
            out_deepstack[ds_idx].append(ds_comp)
        offset += num

    tokens_comp = torch.cat(out_tokens, dim=0) if len(out_tokens) > 0 else hidden_states[:0]
    keep_index = (
        torch.cat(out_keep, dim=0).to(torch.long)
        if len(out_keep) > 0
        else hidden_states.new_zeros((0,), dtype=torch.long)
    )
    deepstack_comp = [
        torch.cat(chunks, dim=0) if len(chunks) > 0 else ds[:0]
        for chunks, ds in zip(out_deepstack, deepstack_feature_lists)
    ]
    return tokens_comp, deepstack_comp, keep_index


class Qwen3VLVisionModelVCast(Qwen3VLVisionModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        v_cast_compress = bool(kwargs.pop("v_cast_compress", False))
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        if not v_cast_compress:
            return hidden_states, deepstack_feature_lists

        is_video = grid_thw is not None and torch.any(grid_thw[:, 0] > 1).item()
        if not is_video:
            keep_index = torch.arange(int(hidden_states.shape[0]), device=hidden_states.device, dtype=torch.long)
            return hidden_states, deepstack_feature_lists, keep_index

        tokens_comp, deepstack_comp, keep_index = v_cast_compress_qwen3(
            hidden_states,
            deepstack_feature_lists,
            grid_thw=grid_thw,
            spatial_merge_size=int(getattr(self.config, "spatial_merge_size", 1)),
            retain_ratio=DEFAULT_RETAIN_RATIO,
            min_k=DEFAULT_MIN_K,
        )
        return tokens_comp, deepstack_comp, keep_index


class Qwen3VLModelVCast(Qwen3VLModel):
    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        visual_out = self.visual(pixel_values_videos, grid_thw=video_grid_thw, v_cast_compress=True)

        if not isinstance(visual_out, tuple):
            raise ValueError(f"[V-CAST] Unexpected visual output type: {type(visual_out)}")

        if len(visual_out) == 3:
            video_embeds, deepstack_video_embeds, keep_index_placeholder = visual_out
            return video_embeds, deepstack_video_embeds, keep_index_placeholder

        if len(visual_out) == 2:
            video_embeds, deepstack_video_embeds = visual_out
            keep_index_placeholder = torch.arange(
                int(video_embeds.shape[0]),
                device=video_embeds.device,
                dtype=torch.long,
            )
            if _v_cast_print() and not getattr(self, "_v_cast_warned_visual_return_2", False):
                print("[V-CAST] visual returned 2 values; fallback to keep-all video indices (MoE compatibility mode).")
                self._v_cast_warned_visual_return_2 = True
            return video_embeds, deepstack_video_embeds, keep_index_placeholder

        raise ValueError(f"[V-CAST] Unexpected visual output tuple length: {len(visual_out)}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        is_prefill = False
        if input_ids is not None and input_ids.shape[1] != 1:
            is_prefill = True
        elif cache_position is not None and cache_position[0] == 0:
            is_prefill = True
        elif self.rope_deltas is None:
            is_prefill = True

        if position_ids is None:
            if is_prefill:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                if self.rope_deltas is None:
                    self.rope_deltas = torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=input_ids.dtype)
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None and delta.ndim == 2 and delta.shape[0] == 1 and batch_size > 1:
                    delta = delta.expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        deepstack_video_embeds = None
        if pixel_values_videos is not None and is_prefill:
            if input_ids is not None and input_ids.shape[0] > 1:
                if _v_cast_print():
                    print("[V-CAST] batch>1 detected; disabling compression and pruning for safety.")
                video_embeds, deepstack_video_embeds_list = self.visual(
                    pixel_values_videos, grid_thw=video_grid_thw, v_cast_compress=False
                )
                deepstack_video_embeds = deepstack_video_embeds_list
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            else:
                video_embeds, deepstack_video_embeds_list, keep_index_placeholder = self.get_video_features(
                    pixel_values_videos,
                    video_grid_thw,
                )
                deepstack_video_embeds = deepstack_video_embeds_list

                video_token_id = self.config.video_token_id
                video_indices_in_input = (input_ids == video_token_id).nonzero(as_tuple=True)

                tokens_to_keep_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
                total_video_tokens = int(video_indices_in_input[0].shape[0])
                if total_video_tokens > 0:
                    keep_index_placeholder = keep_index_placeholder.to(input_ids.device)
                    safe_idx = keep_index_placeholder.clamp(min=0, max=total_video_tokens - 1)
                    visual_keep_mask = torch.zeros(total_video_tokens, dtype=torch.bool, device=input_ids.device)
                    visual_keep_mask[safe_idx] = True
                    rows_to_drop = video_indices_in_input[0][~visual_keep_mask]
                    cols_to_drop = video_indices_in_input[1][~visual_keep_mask]
                    tokens_to_keep_mask[rows_to_drop, cols_to_drop] = False

                if input_ids.shape[0] == 1:
                    mask_1d = tokens_to_keep_mask[0]
                    input_ids = input_ids[:, mask_1d]
                    inputs_embeds = inputs_embeds[:, mask_1d]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, mask_1d]
                    if position_ids is not None:
                        position_ids = position_ids[:, :, mask_1d]
                        if self.rope_deltas is not None and self.rope_deltas.shape[1] == mask_1d.shape[0]:
                            if self.rope_deltas.dim() == 2:
                                self.rope_deltas = self.rope_deltas[:, mask_1d]
                            elif self.rope_deltas.dim() == 3:
                                self.rope_deltas = self.rope_deltas[:, mask_1d, :]

                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds_final = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds_final = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds_final.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds_final = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds_final = deepstack_video_embeds

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds_final,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

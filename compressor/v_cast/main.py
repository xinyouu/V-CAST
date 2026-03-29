import types

from .modeling_qwen3_vl_v_cast import (
    Qwen3VLModelVCast,
    Qwen3VLVisionModelVCast,
)


def v_cast(model):
    print("################################")
    print("############ V-CAST ############")
    print("################################")
    print("[V-CAST] retain_ratio=0.25 budget=curvature/softmax@0.7 score=hybrid keep_ends=False")

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLVisionModel,
    )

    Qwen3VLVisionModel.forward = Qwen3VLVisionModelVCast.forward
    Qwen3VLModel.get_video_features = Qwen3VLModelVCast.get_video_features
    Qwen3VLModel.forward = Qwen3VLModelVCast.forward

    # MoE checkpoints (e.g. Qwen3-VL-30B-A3B) use classes from qwen3_vl_moe.
    # Patch vision backbones there as well, otherwise v_cast_compress=True is ignored.
    patched_moe_visual = []
    try:
        import importlib

        moe_modeling = importlib.import_module("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
    except Exception:
        moe_modeling = None

    if moe_modeling is not None:
        for _, obj in vars(moe_modeling).items():
            if not isinstance(obj, type):
                continue
            cls_name = getattr(obj, "__name__", str(obj))
            # MoE vision class naming can vary by transformers version.
            # Match by class name first, then by minimal structural traits.
            looks_like_vision_backbone = (
                ("VisionModel" in cls_name and hasattr(obj, "forward"))
                or (
                    hasattr(obj, "forward")
                    and hasattr(obj, "patch_embed")
                    and hasattr(obj, "blocks")
                )
            )
            if not looks_like_vision_backbone:
                continue
            setattr(obj, "forward", Qwen3VLVisionModelVCast.forward)
            patched_moe_visual.append(cls_name)

    if patched_moe_visual:
        names = ", ".join(sorted(set(patched_moe_visual)))
        print(f"[V-CAST] Patched MoE vision classes: {names}")

    # Also patch the current runtime visual instance directly.
    # This avoids class-resolution gaps across different transformers versions.
    try:
        backbone = getattr(model, "model", None)
        visual = getattr(backbone, "visual", None) if backbone is not None else None
        if visual is not None and callable(getattr(visual, "forward", None)):
            visual.forward = types.MethodType(Qwen3VLVisionModelVCast.forward, visual)
            print(f"[V-CAST] Patched runtime visual instance: {type(visual).__name__}")
    except Exception as exc:
        print(f"[V-CAST] Warning: failed to patch runtime visual instance: {exc}")

    return model

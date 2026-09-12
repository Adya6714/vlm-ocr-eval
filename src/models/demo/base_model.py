"""
Load a pretrained document VLM for the Stage 2b demo.

Why this exists: the demo is a production-shaped stack (Decision #1),
not the exposure instrument. It starts from a pretrained checkpoint so
LoRA SFT can look like how a real system is actually built (BOOK.md
Chapter 4).

Which checkpoint: Decision #3 is still the T4 VRAM measurement. This
file's DEFAULT_MODEL_ID is a *development default* for wiring SFT, not
a closed T4 call — see DECISIONS.md #79 and docs/tier2_stage2b_demo.md.

Called from: lora_config.py / sft.py / benchmark_base_models.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Development default only. Do not treat as "fits on T4 with headroom."
DEFAULT_MODEL_ID = "ds4sd/SmolDocling-256M-preview"

CANDIDATES = {
    "smoldocling": "ds4sd/SmolDocling-256M-preview",
    "granite-docling": "ibm-granite/granite-docling-258M",
    "lightonocr-1b": "lightonai/LightOnOCR-1B-1025",
    "lightonocr-2-1b": "lightonai/LightOnOCR-2-1B",
    "lightonocr-2-1b-base": "lightonai/LightOnOCR-2-1B-base",
}


def load_model_and_processor(model_id: str, torch_dtype=None):
    """
    Load processor + causal VLM. LightOnOCR prefers its dedicated class
    when transformers still ships it; otherwise AutoModelForImageTextToText
    (transformers 5+) / AutoModelForVision2Seq.
    """
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if "lightonai" in model_id.lower() or "lightonocr" in model_id.lower():
        try:
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

            processor = LightOnOcrProcessor.from_pretrained(model_id)
            model = LightOnOcrForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch_dtype
            )
            return model, processor
        except ImportError:
            # transformers>=5 dropped the dedicated class on some builds;
            # fall through to Auto*.
            pass
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM
    model = _AutoVLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    return model, processor


def inspect_leaf_attn_names(model) -> list[str]:
    """
    Unique last path segments of PEFT-wrappable layers (Linear etc.).

    Not container names such as self_attn: PEFT matches by substring and
    can only wrap Linear/Embedding/Conv/MultiheadAttention.
    """
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from benchmark_base_models import collect_peft_target_leaves

    leaves, _ = collect_peft_target_leaves(model)
    return leaves


def main() -> None:
    ap = argparse.ArgumentParser(description="Load/inspect a demo backbone")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    model, _ = load_model_and_processor(args.model_id, torch_dtype=torch.float32)
    if args.inspect:
        print(f"[inspect] {args.model_id} type={type(model).__name__}")
        for leaf in inspect_leaf_attn_names(model):
            print(f"  leaf {leaf}")
        print("[inspect] full dotted names containing attn/proj:")
        for name, _ in model.named_modules():
            if "proj" in name or "attn" in name.lower():
                print(f"  {name}")


if __name__ == "__main__":
    main()

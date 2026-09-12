"""
Stage 2b prerequisite: benchmark candidate base VLMs for T4 memory fit.
Answers DECISIONS.md #3, currently "Not yet benchmarked."

Two modes:
  --inspect   just print attention-related module names (needed to set
              LoRA's target_modules correctly per model -- do NOT guess
              this, run inspect first)
  (default)   wrap with LoRA at the given target_modules, run a few
              dummy train steps, report peak VRAM

Model IDs (confirmed real HF repos, not guessed):
  SmolDocling: ds4sd/SmolDocling-256M-preview
  LightOnOCR:  lightonai/LightOnOCR-1B-1025 (needs bleeding-edge
               transformers: pip install git+https://github.com/huggingface/transformers)

Note: both candidates now have newer successors (granite-docling-258M,
LightOnOCR-2-1B) per their own model cards -- not swapped in here,
that's a decision for DECISIONS.md #3, not this script.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
INSPECT_JSON = _REPO / "docs" / "demo_lora_inspect.json"

CANDIDATE_IDS = [
    "ds4sd/SmolDocling-256M-preview",
    "ibm-granite/granite-docling-258M",
    "lightonai/LightOnOCR-1B-1025",
    "lightonai/LightOnOCR-2-1B",
    "lightonai/LightOnOCR-2-1B-base",
]


def load_model_and_processor(model_id: str, torch_dtype=None):
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if "lightonai" in model_id:
        try:
            from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
            processor = LightOnOcrProcessor.from_pretrained(model_id)
            model = LightOnOcrForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch_dtype
            )
            return model, processor
        except ImportError:
            pass
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM
    model = _AutoVLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    return model, processor


def peft_supported_types() -> tuple[type, ...]:
    """
    Classes PEFT will actually wrap. Container modules (self_attn,
    modality_projection, …) are not in this set even when their name
    contains 'attn' or 'proj' — PEFT substring-matches leaf names, so
    listing a container crashes with 'Target module is not supported'.
    """
    types: list[type] = [
        nn.Linear,
        nn.Embedding,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.MultiheadAttention,
    ]
    try:
        from transformers.pytorch_utils import Conv1D

        types.append(Conv1D)
    except ImportError:
        pass
    return tuple(types)


def collect_peft_target_leaves(model) -> tuple[list[str], list[str]]:
    """
    Unique last-path-segment names of PEFT-wrappable modules.

    Returns (sorted leaf names, dotted paths of those modules) so inspect
    json records what was actually wrap-safe, not container names.
    """
    supported = peft_supported_types()
    leaves: set[str] = set()
    dotted: list[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, supported):
            leaf = name.rsplit(".", 1)[-1]
            leaves.add(leaf)
            dotted.append(name)
    # Sanity: every emitted leaf must correspond only to supported types
    # (same leaf name on a container would still break PEFT).
    for name, module in model.named_modules():
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in leaves and not isinstance(module, supported):
            raise RuntimeError(
                f"leaf {leaf!r} also names unsupported {type(module).__name__} "
                f"at {name}; PEFT substring match would wrap the container"
            )
    return sorted(leaves), dotted


def inspect_modules(model_id: str) -> dict:
    # Names only. A CPU/MPS load is not a T4 VRAM measurement.
    model, _ = load_model_and_processor(model_id)
    print(f"[inspect] PEFT-wrappable modules in {model_id}:")
    leaves, dotted = collect_peft_target_leaves(model)
    for name in dotted:
        print(f"  {name}")
    print("[inspect] leaf names for PEFT target_modules:")
    for leaf in leaves:
        print(f"  {leaf}")
    rec = {
        "model_id": model_id,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "type": type(model).__name__,
        "dotted_peft_supported": dotted,
        "target_modules": leaves,
    }
    payload = {}
    if INSPECT_JSON.exists():
        payload = json.loads(INSPECT_JSON.read_text(encoding="utf-8"))
    payload[model_id] = rec
    INSPECT_JSON.parent.mkdir(parents=True, exist_ok=True)
    INSPECT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[inspect] wrote {INSPECT_JSON}")
    return rec


def benchmark(model_id: str, target_modules: list[str], lora_rank: int,
              batch_size: int, image_size: int, seq_len: int, steps: int,
              device_str: str = "cuda") -> dict:
    if device_str == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required for T4 VRAM numbers. Refusing to report MPS/CPU "
            "as if it closed Decision #3."
        )
    from peft import LoraConfig, get_peft_model

    device = torch.device(device_str)
    print(f"[benchmark] loading {model_id} ...")
    t0 = time.time()
    model, _ = load_model_and_processor(model_id)
    model.to(device)
    print(f"[benchmark] base model loaded in {time.time()-t0:.1f}s")

    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2,
                               target_modules=target_modules, lora_dropout=0.05)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dummy_pixel_values = torch.randn(batch_size, 3, image_size, image_size,
                                       dtype=torch.float16, device=device)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    dummy_labels = dummy_input_ids.clone()

    print(f"[benchmark] running {steps} dummy train steps, batch_size={batch_size} ...")
    for step in range(steps):
        optimizer.zero_grad()
        out = model(pixel_values=dummy_pixel_values, input_ids=dummy_input_ids, labels=dummy_labels)
        out.loss.backward()
        optimizer.step()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[benchmark] step {step+1}/{steps} loss={out.loss.item():.3f} peak_mem={peak_gb:.2f}GB")

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    fits = peak_gb < 14
    print(f"[benchmark] === {model_id} === peak VRAM: {peak_gb:.2f}GB at batch_size={batch_size}")
    print(f"[benchmark] T4 has 16GB -- {'FITS' if fits else 'TOO TIGHT / OOM RISK'} (14GB threshold leaves headroom)")
    return {
        "model_id": model_id,
        "peak_gb": peak_gb,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "target_modules": list(target_modules),
        "fits_t4_14gb_headroom": fits,
        "t4_gb": 16,
        "headroom_gb": 16.0 - peak_gb,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True,
                     help="ds4sd/SmolDocling-256M-preview or lightonai/LightOnOCR-1B-1025")
    ap.add_argument("--inspect", action="store_true", help="print module names, don't train")
    ap.add_argument("--target-modules", nargs="+", default=None,
                     help="required unless --inspect; get real names from --inspect first")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--json-out",
        default=None,
        help="Write peak-VRAM dict here (Decision #3 evidence)",
    )
    args = ap.parse_args()

    if args.inspect:
        inspect_modules(args.model_id)
        return
    if not args.target_modules:
        raise SystemExit("--target-modules required unless --inspect (run --inspect first)")
    rec = benchmark(args.model_id, args.target_modules, args.lora_rank, args.batch_size,
              args.image_size, args.seq_len, args.steps, args.device)
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        if p.exists():
            payload = json.loads(p.read_text(encoding="utf-8"))
        payload[args.model_id] = rec
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[benchmark] wrote {p}")


if __name__ == "__main__":
    main()

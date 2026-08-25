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
import time

import torch


def load_model_and_processor(model_id: str):
    if "lightonai" in model_id:
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        processor = LightOnOcrProcessor.from_pretrained(model_id)
        model = LightOnOcrForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)
    return model, processor


def inspect_modules(model_id: str) -> None:
    model, _ = load_model_and_processor(model_id)
    print(f"[inspect] attention/projection module names in {model_id}:")
    for name, _ in model.named_modules():
        if "proj" in name or "attn" in name.lower():
            print(f"  {name}")
    print("[inspect] pick the real q_proj/v_proj-equivalent names above for --target-modules")


def benchmark(model_id: str, target_modules: list[str], lora_rank: int,
              batch_size: int, image_size: int, seq_len: int, steps: int,
              device_str: str = "cuda") -> None:
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
    print(f"[benchmark] === {model_id} === peak VRAM: {peak_gb:.2f}GB at batch_size={batch_size}")
    print(f"[benchmark] T4 has 16GB -- {'FITS' if peak_gb < 14 else 'TOO TIGHT / OOM RISK'} (14GB threshold leaves headroom)")


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
    args = ap.parse_args()

    if args.inspect:
        inspect_modules(args.model_id)
        return
    if not args.target_modules:
        raise SystemExit("--target-modules required unless --inspect (run --inspect first)")
    benchmark(args.model_id, args.target_modules, args.lora_rank, args.batch_size,
              args.image_size, args.seq_len, args.steps, args.device)


if __name__ == "__main__":
    main()

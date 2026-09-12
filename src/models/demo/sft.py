"""
Supervised LoRA fine-tune for the demo VLM (Stage 2b).

Why this training set: `data/manifests/hindi_natural.jsonl` line crops.
That is this project's OCR data with exact GT, already Colab-portable
via --data-root. It is *not* an exposure-controlled experiment (Decision
#1: the demo is allowed pretrained Indic knowledge). A broader GlotOCR
page-level set would be more "production-like" but has no layout tags
and mixes scripts; flagged in docs/tier2_stage2b_demo.md.

Resume: checkpoint every N steps under output-root; skip-if-complete
like the instrument trainer. fp16 + grad checkpoint on CUDA; this file
will not pretend to have run if no GPU / no inspect json.

Does not start training unless --run is passed *and* a CUDA device
exists, so importing this module on a laptop is safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from lora_config import build_lora_config
from base_model import DEFAULT_MODEL_ID, load_model_and_processor


class LineCropDataset(Dataset):
    """Manifest rows `{"image_path","text"}` relative to data-root/repo."""

    def __init__(self, manifest: Path, data_root: Path):
        self.rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        path = Path(row["image_path"])
        if not path.is_absolute():
            cand = self.data_root / path
            if not cand.exists():
                cand = self.data_root.parent / path
            path = cand
        image = Image.open(path).convert("RGB")
        return {"image": image, "text": row["text"], "image_path": str(path)}


def checkpoint_path(output_root: Path, step: int) -> Path:
    return output_root / f"demo_lora_step{step}.pt"


def train(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "SFT --run requires CUDA (T4). This host has no CUDA; "
            "not substituting MPS/CPU as a T4 result."
        )
    from peft import get_peft_model

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model, processor = load_model_and_processor(args.model_id)
    lora = build_lora_config(args.model_id, r=args.lora_rank)
    model = get_peft_model(model, lora)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(args.device)
    model.train()
    ds = LineCropDataset(Path(args.manifest), Path(args.data_root))
    if len(ds) == 0:
        raise SystemExit(f"empty manifest {args.manifest}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    meta = {
        "model_id": args.model_id,
        "n_lines": len(ds),
        "max_steps": args.max_steps,
        "lora_rank": args.lora_rank,
    }
    (output_root / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    start = 0
    state_path = output_root / "sft_state.json"
    if state_path.exists():
        start = int(json.loads(state_path.read_text())["step"])
        print(f"[sft] resuming from step {start}")
    print(f"[sft] n_lines={len(ds)} max_steps={args.max_steps} device={args.device}")
    for step in range(start, args.max_steps):
        item = ds[step % len(ds)]
        try:
            batch = _encode_example(processor, item, args.device)
        except Exception as e:
            marker = {
                "model_id": args.model_id,
                "step": step,
                "reason": "processor_collate_failed",
                "error": f"{type(e).__name__}: {e}",
            }
            (output_root / "sft_not_trained.json").write_text(
                json.dumps(marker, indent=2), encoding="utf-8"
            )
            print(f"[sft] collate failed at step {step}: {marker['error']}")
            print("[sft] stopping. Partial: LoRA wrap only, no adapter save.")
            return
        opt.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        opt.step()
        print(f"[sft] step {step+1}/{args.max_steps} loss={float(loss.item()):.4f}")
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.max_steps:
            ckpt_dir = output_root / f"step_{step+1}"
            model.save_pretrained(ckpt_dir)
            model.save_pretrained(output_root)
            state_path.write_text(json.dumps({"step": step + 1}), encoding="utf-8")
            print(f"[sft] checkpoint {ckpt_dir}")
    print(f"[sft] finished {args.max_steps} steps; adapter in {output_root}")


def _encode_example(processor, item: dict, device: str) -> dict:
    """
    Best-effort image+text collate. Chat-template models differ; if this
    raises, sft.py stops rather than training on dummy tensors.
    """
    image = item["image"]
    text = item["text"]
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Transcribe the text in this image."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        enc = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    else:
        enc = processor(images=image, text=text, return_tensors="pt", padding=True)
    if "labels" not in enc:
        enc["labels"] = enc.get("input_ids")
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo LoRA SFT")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument(
        "--manifest",
        default="data/manifests/hindi_natural.jsonl",
        help="Line-crop JSONL with exact GT (design choice: project OCR data)",
    )
    ap.add_argument("--data-root", default=os.environ.get("OCR_DATA_ROOT", "data"))
    ap.add_argument("--output-root", default="checkpoints/demo")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--run",
        action="store_true",
        help="Attempt training. Refuses without CUDA.",
    )
    args = ap.parse_args()
    if args.run:
        train(args)
    else:
        print("[sft] dry. pass --run on a T4 after inspect json exists.")


if __name__ == "__main__":
    main()

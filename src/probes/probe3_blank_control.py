"""
Probe 3 — blank/noise-image control.

Question: does the model's output/confidence come from actually reading
the image, or from language-prior guessing? Feed it three conditions per
real line-crop (real image, blank, matched noise) and compare confidence
and text. If blank/noise confidence tracks real-image confidence, the
model isn't reading — it's guessing from what the language "usually
looks like."
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# probe3_blank_control.py lives in src/probes/, but generate.py /
# train.py live in src/models/instrument/ — same sys.path pattern as
# probe1_exposure.py so imports work regardless of cwd.
_INSTRUMENT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "instrument")
)
sys.path.insert(0, _INSTRUMENT_DIR)

from generate import generate
from tokenizer import GraphemeTokenizer
from train import InstrumentModel, PATCH_SIZE


def load_model_and_tokenizer(output_root: Path, condition: str, seed: int, device: torch.device):
    tokenizer_path = output_root / f"tokenizer_{condition}.json"
    ckpt_path = output_root / f"checkpoint_{condition}_seed{seed}.pt"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"no tokenizer at {tokenizer_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    tokenizer = GraphemeTokenizer.load(str(tokenizer_path))
    model = InstrumentModel(vocab_size=len(tokenizer))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def prepare_image_tensor(image: Image.Image, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Mirrors collate_batch's preprocessing for a single image. Once
    export_line_crops produces a FIXED canonical height (Cursor task
    below), this only needs to pad width, same as training. Left
    general (pads both dims) so it still works even if some crops
    predate the fix.
    """
    if image.mode != "L":
        image = image.convert("L")
    w, h = image.size
    pad_w = ((w + patch_size - 1) // patch_size) * patch_size - w
    pad_h = ((h + patch_size - 1) // patch_size) * patch_size - h
    arr = torch.from_numpy(np.array(image, dtype="float32") / 255.0).unsqueeze(0)  # [1, H, W]
    arr = torch.nn.functional.pad(arr, (0, pad_w, 0, pad_h), value=1.0)  # white pad
    return arr.unsqueeze(0)  # [1, 1, H, W]


def make_blank(image: Image.Image) -> Image.Image:
    return Image.new("L", image.size, color=255)


def make_matched_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.array(image, dtype="float32")
    mean, std = arr.mean(), arr.std()
    noise = rng.normal(loc=mean, scale=max(std, 1.0), size=arr.shape)
    noise = np.clip(noise, 0, 255).astype("uint8")
    return Image.fromarray(noise, mode="L")


def run_probe3(manifest_path: Path, output_root: Path, condition: str, seed: int,
                n_samples: int, out_path: Path, device_str: str = "cpu") -> None:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(output_root, condition, seed, device)

    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    sample_rows = random.Random(0).sample(rows, min(n_samples, len(rows)))
    rng_np = np.random.default_rng(0)

    results = []
    for i, row in enumerate(sample_rows):
        real_img = Image.open(row["image_path"])
        blank_img = make_blank(real_img)
        noise_img = make_matched_noise(real_img, rng_np)

        entry = {"ground_truth": row["text"], "image_path": row["image_path"]}
        for cond_name, img in [("real", real_img), ("blank", blank_img), ("noise", noise_img)]:
            tensor = prepare_image_tensor(img).to(device)
            out = generate(model, tensor, tokenizer)
            entry[cond_name] = {
                "text": out["text"],
                "mean_confidence": float(np.mean(out["step_confidences"])) if out["step_confidences"] else None,
            }
        results.append(entry)
        if i % 10 == 0:
            print(f"[probe3] {i+1}/{len(sample_rows)} done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name in ("real", "blank", "noise"):
        vals = [r[name]["mean_confidence"] for r in results if r[name]["mean_confidence"] is not None]
        print(f"[probe3] mean confidence ({name}): {np.mean(vals):.4f}")
    print(f"[probe3] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--condition", required=True, choices=["natural", "flattened", "inverted"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_probe3(Path(args.manifest), Path(args.output_root), args.condition, args.seed,
                args.n_samples, Path(args.out), args.device)


if __name__ == "__main__":
    main()

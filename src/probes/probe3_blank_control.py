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
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from probe_utils import load_model_and_tokenizer, run_generate


def make_blank(image: Image.Image) -> Image.Image:
    return Image.new("L", image.size, color=255)


def make_matched_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.array(image, dtype="float32")
    mean, std = arr.mean(), arr.std()
    noise = rng.normal(loc=mean, scale=max(std, 1.0), size=arr.shape)
    noise = np.clip(noise, 0, 255).astype("uint8")
    return Image.fromarray(noise, mode="L")


def run_probe3(manifest_path: Path, output_root: Path, script: str, condition: str, seed: int,
                n_samples: int, out_path: Path, device_str: str = "cpu") -> None:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(output_root, script, condition, seed, device)

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
            out = run_generate(model, tokenizer, img, device)
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
    ap.add_argument("--script", required=True, choices=["hindi", "bengali"])
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--condition", required=True, choices=["natural", "flattened", "inverted"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_probe3(Path(args.manifest), Path(args.output_root), args.script, args.condition, args.seed,
                args.n_samples, Path(args.out), args.device)


if __name__ == "__main__":
    main()

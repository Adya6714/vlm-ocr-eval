"""
Generates instrument predictions for image/text pairs, writing the
{"prediction","ground_truth"} shape Probe 6 expects. Handles both:
  - Tier C real ground_truth.jsonl (keys: img_plain_path, text)
  - Synthetic manifests (keys: image_path, text)
Real images are resized to canonical height first; synthetic line
crops are already at that height, so resize is a no-op for them.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image

from probe_utils import load_model_and_tokenizer, run_generate, resize_to_canonical_height


def run(manifest_path: Path, image_key: str, text_key: str, output_root: Path,
        condition: str, seed: int, n_samples: int, out_path: Path,
        device_str: str = "cpu") -> None:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(output_root, condition, seed, device)

    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    if n_samples and n_samples < len(rows):
        rows = random.Random(0).sample(rows, n_samples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            img = resize_to_canonical_height(Image.open(row[image_key]))
            out = run_generate(model, tokenizer, img, device)
            f.write(json.dumps({"prediction": out["text"], "ground_truth": row[text_key]},
                                 ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"[gen_predictions] {i+1}/{len(rows)} done")
    print(f"[gen_predictions] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-key", default="image_path")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-samples", type=int, default=0, help="0 = use all rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(Path(args.manifest), args.image_key, args.text_key, Path(args.output_root),
        args.condition, args.seed, args.n_samples, Path(args.out), args.device)


if __name__ == "__main__":
    main()

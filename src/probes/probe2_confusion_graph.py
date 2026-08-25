"""
Probe 2, confusion graph.

At each generation step, step_top_k holds the top-5 candidates. This
probe aggregates, across many real images, how often each runner-up
candidate shows up behind the chosen token, weighted by the runner-up's
own probability -- per generate.py's own docstring: "not just what the
model said, but what it almost said instead." No ground-truth alignment
needed, this is purely about the model's own output distribution.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from probe_utils import load_model_and_tokenizer, run_generate


def accumulate_confusions(step_top_k: list, edges: dict) -> None:
    if not step_top_k:
        return
    chosen, _ = step_top_k[0]
    for runner_up, prob in step_top_k[1:]:
        if runner_up == chosen:
            continue
        edges[(chosen, runner_up)] += prob


def run_probe2(manifest_path: Path, output_root: Path, condition: str, seed: int,
                n_samples: int, out_path: Path, device_str: str = "cpu") -> None:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(output_root, condition, seed, device)

    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    sample_rows = random.Random(0).sample(rows, min(n_samples, len(rows)))

    edges: dict[tuple[str, str], float] = defaultdict(float)
    step_count = 0
    for i, row in enumerate(sample_rows):
        img = Image.open(row["image_path"])
        out = run_generate(model, tokenizer, img, device)
        for step_top_k in out["step_top_k"]:
            accumulate_confusions(step_top_k, edges)
            step_count += 1
        if i % 10 == 0:
            print(f"[probe2] {i+1}/{len(sample_rows)} done")

    sorted_edges = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)
    records = [{"chosen": a, "confused_with": b, "weight": w} for (a, b), w in sorted_edges]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"edges": records, "n_steps": step_count, "n_samples": len(sample_rows)},
                   f, ensure_ascii=False, indent=2)

    print(f"[probe2] wrote {out_path} ({len(records)} confusion pairs from {step_count} steps)")
    print("[probe2] top 10 confusion pairs:")
    for rec in records[:10]:
        print(f"  {rec['chosen']!r} <-> {rec['confused_with']!r}: weight={rec['weight']:.3f}")


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
    run_probe2(Path(args.manifest), Path(args.output_root), args.condition, args.seed,
                args.n_samples, Path(args.out), args.device)


if __name__ == "__main__":
    main()

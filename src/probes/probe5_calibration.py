"""
Probe 5, calibration.

Question: when the model is confident, is it actually more likely to
be right? Bucket real crop predictions by mean step confidence,
compare against actual correctness in each bucket. A well calibrated
model's buckets should track accuracy roughly one to one (a 60 percent
confidence bucket should be about 60 percent correct). Divergence is
the finding.

Correctness uses Stage 0 Tier 1/2 equivalence (not raw string match):
Tier 1 encoding variants count as correct; Tier 2 phonetic match only
for languages in SCRIPT_MAP (hindi, bengali).
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

from probe_utils import load_model_and_tokenizer, run_generate

# Same pattern as probe1_exposure.py, but targeting src/eval/ so we can
# import Tier 1/2 scorers as bare siblings (eval/ has no package
# __init__.py; inserting the folder itself matches how probe1 inserts
# models/instrument/ then does `from train import ...`).
_EVAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "eval")
)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from equivalence_tables import tier1_equivalent
from transliteration_equivalence import tier2_equivalent, SCRIPT_MAP


def is_correct(prediction: str, ground_truth: str, language: str) -> bool:
    """
    Real Stage 0 methodology: exact or Tier 1 (encoding-equivalent)
    counts as correct outright. Tier 2 (transliteration-equivalent)
    only applies for languages in SCRIPT_MAP (hindi, bengali) --
    santhali/kashmiri fall through to False rather than raising,
    since to_canonical() would KeyError on them.
    """
    if tier1_equivalent(prediction, ground_truth):
        return True
    if language in SCRIPT_MAP:
        return tier2_equivalent(ground_truth, prediction, language)
    return False


def run_probe5(manifest_path: Path, output_root: Path, condition: str, seed: int,
                language: str, n_samples: int, out_path: Path, device_str: str = "cpu") -> None:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(output_root, condition, seed, device)

    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines()]
    sample_rows = random.Random(0).sample(rows, min(n_samples, len(rows)))

    records = []
    for i, row in enumerate(sample_rows):
        img = Image.open(row["image_path"])
        out = run_generate(model, tokenizer, img, device)
        conf = float(np.mean(out["step_confidences"])) if out["step_confidences"] else 0.0
        correct = is_correct(out["text"], row["text"], language)
        records.append({"confidence": conf, "correct": correct,
                         "prediction": out["text"], "ground_truth": row["text"]})
        if i % 10 == 0:
            print(f"[probe5] {i+1}/{len(sample_rows)} done")

    bins = np.linspace(0, 1, 11)
    bucket_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bucket = [r for r in records if lo <= r["confidence"] < hi]
        if not in_bucket:
            continue
        acc = sum(r["correct"] for r in in_bucket) / len(in_bucket)
        bucket_stats.append({"range": [round(lo, 2), round(hi, 2)], "n": len(in_bucket), "accuracy": acc})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"records": records, "buckets": bucket_stats}, f, ensure_ascii=False, indent=2)

    print(f"[probe5] wrote {out_path}")
    for b in bucket_stats:
        print(f"  conf {b['range']}: acc={b['accuracy']:.2f} (n={b['n']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--condition", required=True, choices=["natural", "flattened", "inverted"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--language", required=True, choices=list(SCRIPT_MAP) + ["santhali", "kashmiri"])
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_probe5(Path(args.manifest), Path(args.output_root), args.condition, args.seed,
                args.language, args.n_samples, Path(args.out), args.device)


if __name__ == "__main__":
    main()

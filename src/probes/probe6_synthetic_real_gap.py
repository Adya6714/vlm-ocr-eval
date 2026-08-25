"""
Probe 6 -- synthetic-to-real accuracy gap, metric-only, per system.

Compares each system's accuracy on Tier C (real) against the same
system's accuracy on Tier A/B (synthetic), reports the gap per system.
Reuses Stage 0's equivalence methodology (tier1/tier2) so real and
synthetic numbers are computed identically and genuinely comparable.

This script only aggregates already-computed predictions -- it does
not run any OCR engine or generate new predictions itself. Both real
and synthetic prediction files must already exist in the same JSONL
shape: one record per line with "prediction" and "ground_truth" keys.

PREREQUISITES NOT YET MET (see chat, do not run for real numbers yet):
  1. No system has synthetic-data predictions yet -- baselines have
     only run against Tier C real data so far.
  2. The instrument's ability to process Tier C's 254px real images
     (vs its 70px training height) is unconfirmed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Same pattern as probe5_calibration.py: insert src/eval/ so Tier 1/2
# scorers import as bare siblings (eval/ has no package __init__.py).
_EVAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "eval")
)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from equivalence_tables import tier1_equivalent
from transliteration_equivalence import tier2_equivalent, SCRIPT_MAP


def is_correct(prediction: str, ground_truth: str, language: str) -> bool:
    if tier1_equivalent(prediction, ground_truth):
        return True
    if language in SCRIPT_MAP:
        return tier2_equivalent(ground_truth, prediction, language)
    return False


def compute_accuracy(predictions_path: Path, language: str) -> float:
    rows = [json.loads(l) for l in predictions_path.read_text(encoding="utf-8").splitlines()]
    correct = sum(is_correct(r["prediction"], r["ground_truth"], language) for r in rows)
    return correct / len(rows) if rows else 0.0


def run_probe6(systems: dict[str, dict[str, Path]], language: str, out_path: Path) -> None:
    """systems: {system_name: {"real": path, "synthetic": path}}"""
    results = {}
    for system, paths in systems.items():
        real_acc = compute_accuracy(paths["real"], language)
        synth_acc = compute_accuracy(paths["synthetic"], language)
        results[system] = {
            "real_accuracy": real_acc,
            "synthetic_accuracy": synth_acc,
            "gap": synth_acc - real_acc,
        }
        print(f"{system:12s} real={real_acc:.3f}  synthetic={synth_acc:.3f}  gap={synth_acc - real_acc:+.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True)
    ap.add_argument("--system", action="append", required=True,
                     help="name:real_path:synthetic_path, repeatable per system")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    systems = {}
    for spec in args.system:
        name, real_path, synth_path = spec.split(":")
        systems[name] = {"real": Path(real_path), "synthetic": Path(synth_path)}

    run_probe6(systems, args.language, Path(args.out))


if __name__ == "__main__":
    main()

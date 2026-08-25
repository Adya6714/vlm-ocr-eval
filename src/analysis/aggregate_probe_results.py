"""
Aggregates Probe 3 + Probe 5 results across all condition/seed runs
for one script (hindi or bengali) into one comparison table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ["natural", "flattened", "inverted"]
SEEDS = [0, 1, 2]


def aggregate_probe3(results_dir: Path, script: str) -> dict:
    summary = {}
    for condition in CONDITIONS:
        real_vals, blank_vals, noise_vals = [], [], []
        for seed in SEEDS:
            path = results_dir / f"probe3_{script}_{condition}_seed{seed}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                real_vals.append(r["real"]["mean_confidence"])
                blank_vals.append(r["blank"]["mean_confidence"])
                noise_vals.append(r["noise"]["mean_confidence"])
        if not real_vals:
            continue
        summary[condition] = {
            "n": len(real_vals),
            "real_conf": float(np.mean(real_vals)),
            "blank_conf": float(np.mean(blank_vals)),
            "noise_conf": float(np.mean(noise_vals)),
            "real_minus_blank_gap": float(np.mean(real_vals) - np.mean(blank_vals)),
        }
    return summary


def aggregate_probe5(results_dir: Path, script: str) -> dict:
    summary = {}
    for condition in CONDITIONS:
        all_records = []
        for seed in SEEDS:
            path = results_dir / f"probe5_{script}_{condition}_seed{seed}.jsonl"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            all_records.extend(data["records"])
        if not all_records:
            continue
        accs = [r["correct"] for r in all_records]
        confs = [r["confidence"] for r in all_records]
        summary[condition] = {
            "n": len(all_records),
            "overall_accuracy": float(np.mean(accs)),
            "mean_confidence": float(np.mean(confs)),
            "confidence_minus_accuracy_gap": float(np.mean(confs) - np.mean(accs)),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/probe_results")
    ap.add_argument("--script", required=True, choices=["hindi", "bengali"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    p3 = aggregate_probe3(results_dir, args.script)
    p5 = aggregate_probe5(results_dir, args.script)

    print(f"\n=== Probe 3 (blank/noise control) — {args.script} ===")
    for cond, s in p3.items():
        print(f"  {cond:10s} n={s['n']:3d}  real={s['real_conf']:.4f}  "
              f"blank={s['blank_conf']:.4f}  noise={s['noise_conf']:.4f}  "
              f"gap(real-blank)={s['real_minus_blank_gap']:+.4f}")

    print(f"\n=== Probe 5 (calibration) — {args.script} ===")
    for cond, s in p5.items():
        print(f"  {cond:10s} n={s['n']:3d}  accuracy={s['overall_accuracy']:.3f}  "
              f"mean_conf={s['mean_confidence']:.3f}  "
              f"overconfidence_gap={s['confidence_minus_accuracy_gap']:+.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"probe3": p3, "probe5": p5}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

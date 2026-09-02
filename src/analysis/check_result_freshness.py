"""
Checks whether a probe result file is stale relative to the checkpoint
it was generated from -- compares the checkpoint's mtime against the
probe result file's recorded generation time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def check_freshness(probe_result_path: Path, checkpoint_path: Path) -> None:
    if not probe_result_path.exists():
        print(f"MISSING: {probe_result_path}")
        return
    if not checkpoint_path.exists():
        print(f"MISSING CHECKPOINT: {checkpoint_path}")
        return
    result_mtime = probe_result_path.stat().st_mtime
    ckpt_mtime = checkpoint_path.stat().st_mtime
    if ckpt_mtime > result_mtime:
        print(f"STALE: {probe_result_path.name} predates {checkpoint_path.name} "
              f"(checkpoint retrained since this result was generated)")
    else:
        print(f"OK: {probe_result_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/probe_results")
    ap.add_argument("--checkpoints-dir", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.checkpoints_dir)
    for f in sorted(results_dir.glob("probe*_*.jsonl")):
        # filenames look like probe3_hindi_natural_seed0.jsonl
        parts = f.stem.split("_")
        script, condition, seed_part = parts[-3], parts[-2], parts[-1]
        ckpt = ckpt_dir / f"checkpoint_{script}_{condition}_{seed_part}.pt"
        check_freshness(f, ckpt)


if __name__ == "__main__":
    main()

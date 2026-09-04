"""
Grapheme-cluster CER / accuracy re-score on saved probe jsonl.

Why this exists: line-level exact / Tier-1/2 `is_correct` collapses to
0.0000 on real Tier C for this instrument, which cannot distinguish
"mostly right" from noise. The paper needs CER at the grapheme-cluster
level (DECISIONS.md #7) on the *already-saved* predictions — no new
inference.

Method: apply `normalize_tier1` to pred and GT, split with Unicode
grapheme clusters (`\\X`), Levenshtein distance / |GT|, accuracy =
max(0, 1 − CER). Tier 2 is *not* applied (CER needs a length-aware
edit metric; Tier 2 is a boolean string gate).

Called from: CLI / paper verification. Reads:
  data/probe_results/probe5_hindi_{natural,flattened,inverted}_seed{N}.jsonl
  data/probe_results/probe5b_hindi_natural_seed{N}.jsonl
  data/probe_results/probe6_synthetic_real_hindi_seed{N}.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import regex

_EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from equivalence_tables import normalize_tier1  # noqa: E402


def grapheme_clusters(text: str) -> list[str]:
    """Unicode grapheme clusters — same unit as DECISIONS.md #7 / #2."""
    return regex.findall(r"\X", text or "")


def levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance over grapheme-cluster sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def cer_and_accuracy(prediction: str, ground_truth: str) -> dict:
    """
    Tier-1-normalized grapheme CER and floored accuracy.

    CER = edit_distance(pred, gt) / len(gt_clusters). Insertions can
    push CER above 1. Accuracy = max(0, 1 − CER). Empty GT: accuracy
    1.0 iff pred also empty, else 0.0.
    """
    pred_c = grapheme_clusters(normalize_tier1(prediction or ""))
    gt_c = grapheme_clusters(normalize_tier1(ground_truth or ""))
    if len(gt_c) == 0:
        empty_match = len(pred_c) == 0
        return {
            "cer": 0.0 if empty_match else float("inf"),
            "accuracy": 1.0 if empty_match else 0.0,
            "n_gt": 0,
            "n_pred": len(pred_c),
            "edit_distance": 0 if empty_match else len(pred_c),
        }
    dist = levenshtein(pred_c, gt_c)
    cer = dist / len(gt_c)
    return {
        "cer": cer,
        "accuracy": max(0.0, 1.0 - cer),
        "n_gt": len(gt_c),
        "n_pred": len(pred_c),
        "edit_distance": dist,
    }


def load_probe_records(path: Path) -> list[dict]:
    """Handle both line-jsonl and probe5's single-object {records, buckets}."""
    text = path.read_text(encoding="utf-8")
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        obj = json.loads(text)
        return obj["records"] if "records" in obj else [obj]
    if len(rows) == 1 and isinstance(rows[0], dict) and "records" in rows[0]:
        return rows[0]["records"]
    return rows


def pred_and_gt(record: dict) -> tuple[str | None, str | None]:
    """Unify probe5 (`prediction`) vs probe5b/6 (`text`) field names."""
    pred = record.get("text")
    if pred is None:
        pred = record.get("prediction")
    gt = record.get("ground_truth")
    if pred is None or gt is None:
        return None, None
    return pred, gt


def summarize(scores: list[dict]) -> dict:
    cers = [s["cer"] for s in scores if s["cer"] != float("inf")]
    accs = [s["accuracy"] for s in scores]
    return {
        "n": len(scores),
        "mean_cer": float(np.mean(cers)) if cers else None,
        "median_cer": float(np.median(cers)) if cers else None,
        "mean_accuracy": float(np.mean(accs)) if accs else None,
        "mean_gt_len": float(np.mean([s["n_gt"] for s in scores])) if scores else None,
    }


def score_file(path: Path, condition: str | None = None) -> list[dict]:
    out = []
    for record in load_probe_records(path):
        if condition is not None and record.get("condition") != condition:
            continue
        pred, gt = pred_and_gt(record)
        if pred is None:
            continue
        stats = cer_and_accuracy(pred, gt)
        stats["condition"] = record.get("condition")
        stats["seed"] = record.get("seed")
        out.append(stats)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--probe-root",
        type=Path,
        default=Path("data/probe_results"),
    )
    args = ap.parse_args()
    root = args.probe_root

    print("=== probe5b (hindi, blank) ===")
    for cond in ("hindi", "blank"):
        pooled: list[dict] = []
        for seed in range(3):
            path = root / f"probe5b_hindi_natural_seed{seed}.jsonl"
            scores = score_file(path, condition=cond)
            pooled.extend(scores)
            st = summarize(scores)
            print(
                f"seed{seed} {cond}: n={st['n']} "
                f"mean_CER={st['mean_cer']:.10f} mean_acc={st['mean_accuracy']:.10f}"
            )
        st = summarize(pooled)
        print(
            f"POOLED {cond}: n={st['n']} "
            f"mean_CER={st['mean_cer']:.10f} mean_acc={st['mean_accuracy']:.10f}"
        )

    print("=== probe6 (real_plain, real_degraded, blank) ===")
    for cond in ("real_plain", "real_degraded", "blank"):
        pooled = []
        for seed in range(3):
            path = root / f"probe6_synthetic_real_hindi_seed{seed}.jsonl"
            scores = score_file(path, condition=cond)
            pooled.extend(scores)
            st = summarize(scores)
            print(
                f"seed{seed} {cond}: n={st['n']} "
                f"mean_CER={st['mean_cer']:.10f} mean_acc={st['mean_accuracy']:.10f}"
            )
        st = summarize(pooled)
        print(
            f"POOLED {cond}: n={st['n']} "
            f"mean_CER={st['mean_cer']:.10f} mean_acc={st['mean_accuracy']:.10f}"
        )

    print("=== probe5 (natural / flattened / inverted) ===")
    for cond in ("natural", "flattened", "inverted"):
        pooled = []
        for seed in range(3):
            path = root / f"probe5_hindi_{cond}_seed{seed}.jsonl"
            scores = score_file(path)
            pooled.extend(scores)
            st = summarize(scores)
            print(
                f"seed{seed} {cond}: n={st['n']} "
                f"mean_CER={st['mean_cer']:.10f} mean_acc={st['mean_accuracy']:.10f}"
            )
        st = summarize(pooled)
        print(
            f"POOLED {cond}: n={st['n']} "
            f"mean_CER={st['mean_cer']:.10f} mean_acc={st['mean_accuracy']:.10f}"
        )


if __name__ == "__main__":
    main()

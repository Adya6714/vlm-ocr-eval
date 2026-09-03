"""
Probe 2 — confusion structure (GT-aligned).

Question: when the instrument misreads a grapheme cluster, was the
correct answer still in the output distribution with real mass, or
near-zero / near-uniform noise? Closed APIs return only the argmax —
owning the model is what makes this probe possible (BOOK.md Ch. 7).

Method:
  1. Greedy-generate on real Hindi Tier C images (same Random(0)
     sample as Probe 5b's hindi condition — DECISIONS.md #57).
  2. Align predicted vs ground-truth grapheme clusters
     (Needleman–Wunsch, DECISIONS.md #7).
  3. At every substitution, record true cluster, predicted cluster,
     top-5 distribution, and the full-softmax probability + rank of
     the TRUE cluster (even when it was not selected).

Checkpoint paths go through probe_utils.load_model_and_tokenizer →
train.checkpoint_path → checkpoint_{script}_{condition}_seed{N}.pt
(DECISIONS.md #47). No pre-#47 bare `checkpoint_natural_seedN.pt`.

Outputs: data/probe_results/probe2_{script}_{condition}_seed{N}.jsonl
Analysis: src/analysis/analyze_probe2.py → docs/probe2_confusion_analysis.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import regex
import torch
from PIL import Image

_PROBES_DIR = Path(__file__).resolve().parent
if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

from probe5b_zeroshot_floor import (  # noqa: E402
    IN_DISTRIBUTION_LANGUAGE,
    load_ground_truth_rows,
    resolve_image_path,
    resolve_repo_root,
)
from probe_utils import (  # noqa: E402
    load_model_and_tokenizer,
    resize_to_canonical_height,
    run_generate,
)
from train import checkpoint_path, tokenizer_path  # noqa: E402  (instrument on path via probe_utils)


SPECIAL = frozenset({"<PAD>", "<BOS>", "<EOS>"})
# <RARE> is a real emitted content token when the model predicts an
# OOV stand-in — keep it in the pred sequence so pred_index still
# lines up with step_top_k / step_probs.


def grapheme_clusters(text: str) -> list[str]:
    """Unicode grapheme clusters — same unit as the instrument vocab."""
    return regex.findall(r"\X", text)


def content_token_clusters(token_ids: list[int], tokenizer) -> list[str]:
    """
    Strip special tokens from a generated id sequence, keep cluster strings.

    Generation step i (0-based) that produced content token j maps
    step_top_k[j] / step_probs[j] → the distribution for that cluster.
    EOS/BOS/PAD/RARE-as-special are excluded so alignment is over
    readable graphemes only.
    """
    out = []
    for tid in token_ids:
        cluster = tokenizer.id_to_cluster.get(int(tid), "<RARE>")
        if cluster in SPECIAL:
            continue
        out.append(cluster)
    return out


def align_substitutions(
    gt_glyphs: list[str],
    pred_glyphs: list[str],
) -> list[dict]:
    """
    Needleman–Wunsch alignment returning only substitution events.

    Why NW and not position-index zip: predictions insert/delete freely,
    so a length mismatch would attribute the wrong distribution step to
    a GT cluster. Match = +1, mismatch/gap = −1 — same costs as
    probe1_fixed_effects._align_glyph_matches (DECISIONS.md #7).

    Each returned dict has true_cluster, predicted_cluster, and
    pred_index (into pred_glyphs / generation content steps).
    Insertions and deletions are ignored — no paired true/pred to score.
    """
    n, m = len(gt_glyphs), len(pred_glyphs)
    if n == 0 or m == 0:
        return []

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up (del), 2 left (ins)
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] - 1
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] - 1
        bt[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1, j - 1] + (1 if gt_glyphs[i - 1] == pred_glyphs[j - 1] else -1)
            delete = dp[i - 1, j] - 1
            insert = dp[i, j - 1] - 1
            best = max(match, delete, insert)
            dp[i, j] = best
            if best == match:
                bt[i, j] = 0
            elif best == delete:
                bt[i, j] = 1
            else:
                bt[i, j] = 2

    subs: list[dict] = []
    i, j = n, m
    while i > 0 and j > 0:
        move = bt[i, j]
        if move == 0:
            gt_c = gt_glyphs[i - 1]
            pred_c = pred_glyphs[j - 1]
            if gt_c != pred_c:
                subs.append({
                    "true_cluster": gt_c,
                    "predicted_cluster": pred_c,
                    "gt_index": i - 1,
                    "pred_index": j - 1,
                })
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    subs.reverse()
    return subs


def true_token_stats(
    probs: torch.Tensor,
    true_cluster: str,
    tokenizer,
) -> dict:
    """
    Probability and 1-based rank of the true cluster under a full softmax.

    If the true cluster is absent from the tokenizer vocab, mass is 0
    and rank is None — the model literally cannot emit it (Probe 5b
    zero-shot case); Probe 2 on in-distribution Hindi should rarely hit
    this, but we record it honestly rather than inventing a rank.
    """
    tid = tokenizer.cluster_to_id.get(true_cluster)
    if tid is None:
        return {
            "true_prob": 0.0,
            "true_rank": None,
            "true_in_vocab": False,
            "true_in_top5": False,
        }
    p = float(probs[tid].item())
    # Rank: 1 = highest prob. Ties broken by stable argsort descending.
    order = torch.argsort(probs, descending=True)
    rank_list = (order == tid).nonzero(as_tuple=False)
    rank = int(rank_list[0].item()) + 1 if rank_list.numel() else None
    top5_ids = set(int(i) for i in order[:5].tolist())
    return {
        "true_prob": p,
        "true_rank": rank,
        "true_in_vocab": True,
        "true_in_top5": tid in top5_ids,
    }


def build_hindi_sample(data_root: Path, repo_root: Path, n_samples: int) -> list[dict]:
    """
    Same Random(0) Hindi Tier C draw as Probe 5b / attention ablation.

    Pool may be smaller than n_samples (currently 60 Hindi GT rows);
    both probes take min(n, pool) so results stay comparable.
    """
    import random

    hindi_rows = load_ground_truth_rows(data_root, IN_DISTRIBUTION_LANGUAGE)
    rng = random.Random(0)
    sample = rng.sample(hindi_rows, min(n_samples, len(hindi_rows)))
    tasks = []
    for row in sample:
        tasks.append({
            "row": row,
            "image_path": str(resolve_image_path(row, repo_root)),
        })
    return tasks


def load_completed_paths(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["image_path"])
    return done


def extract_misreads(
    gen_out: dict,
    ground_truth: str,
    tokenizer,
) -> list[dict]:
    """
    Align generation to GT and attach top-5 + true-token mass per sub.

    Requires gen_out['step_probs'] (full softmax) so true_prob/rank are
    exact, not inferred from top-5 alone — if the correct cluster sits
    at rank 17, top-5 would silently miss it.
    """
    pred_glyphs = content_token_clusters(gen_out["token_ids"], tokenizer)
    gt_glyphs = grapheme_clusters(ground_truth)
    # Drop whitespace-only GT clusters from alignment? Keep them —
    # spaces are real tokens the model may emit; scoring them is fine.
    subs = align_substitutions(gt_glyphs, pred_glyphs)
    step_top_k = gen_out["step_top_k"]
    step_probs = gen_out["step_probs"]
    misreads = []
    for sub in subs:
        j = sub["pred_index"]
        if j >= len(step_probs) or j >= len(step_top_k):
            # EOS-only trailing step or truncated — skip unpaired.
            continue
        stats = true_token_stats(step_probs[j], sub["true_cluster"], tokenizer)
        misreads.append({
            "true_cluster": sub["true_cluster"],
            "predicted_cluster": sub["predicted_cluster"],
            "gt_index": sub["gt_index"],
            "pred_index": j,
            "top5": [
                {"cluster": c, "prob": float(p)} for c, p in step_top_k[j]
            ],
            **stats,
        })
    return misreads


def aggregate_confusion_pairs(misreads: list[dict]) -> list[dict]:
    """
    Collapse misread events into (true, predicted) confusion pairs.

    For each pair: count, mean true_prob, mean true_rank (when defined),
    fraction where true was in top-5. Sorted by count descending —
    the paper's top-15 table comes from the head of this list.
    """
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in misreads:
        buckets[(m["true_cluster"], m["predicted_cluster"])].append(m)

    rows = []
    for (true_c, pred_c), events in buckets.items():
        probs = [e["true_prob"] for e in events]
        ranks = [e["true_rank"] for e in events if e["true_rank"] is not None]
        in_top5 = [e["true_in_top5"] for e in events]
        rows.append({
            "true_cluster": true_c,
            "predicted_cluster": pred_c,
            "count": len(events),
            "mean_true_prob": float(np.mean(probs)) if probs else None,
            "mean_true_rank": float(np.mean(ranks)) if ranks else None,
            "frac_true_in_top5": float(np.mean(in_top5)) if in_top5 else None,
            "example_top5": events[0]["top5"],
        })
    rows.sort(key=lambda r: (-r["count"], r["true_cluster"], r["predicted_cluster"]))
    return rows


def qualitative_tag(true_c: str, pred_c: str) -> str:
    """
    Cheap structural hint for the printed top-15 — not a claim.

    Flags: shared leading code point (same base aksara family), both
    Devanagari, length/matra-ish difference, else 'dissimilar'. A human
    still has to read the pairs; this only steers attention.
    """
    if not true_c or not pred_c:
        return "empty"
    t0, p0 = true_c[0], pred_c[0]
    both_deva = (
        "\u0900" <= t0 <= "\u097F" and "\u0900" <= p0 <= "\u097F"
    )
    if true_c == pred_c:
        return "identical"
    if t0 == p0 and len(true_c) != len(pred_c):
        return "same-base-matra-diff"
    if t0 == p0:
        return "same-base"
    if both_deva and abs(ord(t0) - ord(p0)) <= 5:
        return "adjacent-codepoint"
    if both_deva:
        return "both-devanagari"
    return "dissimilar"


def print_top_pairs(pairs: list[dict], n: int = 15) -> None:
    """Human-facing top-N for Colab logs / qualitative read."""
    print(f"\n[probe2] === top {n} confusion pairs (true → predicted) ===")
    print(
        f"{'rank':>4}  {'n':>4}  {'mean_p(true)':>12}  {'mean_rank':>9}  "
        f"{'in_top5':>7}  {'tag':<22}  pair"
    )
    for i, row in enumerate(pairs[:n], start=1):
        tag = qualitative_tag(row["true_cluster"], row["predicted_cluster"])
        mean_rank = (
            f"{row['mean_true_rank']:.1f}"
            if row["mean_true_rank"] is not None
            else "n/a"
        )
        mean_p = (
            f"{row['mean_true_prob']:.4f}"
            if row["mean_true_prob"] is not None
            else "n/a"
        )
        frac = (
            f"{row['frac_true_in_top5']:.2f}"
            if row["frac_true_in_top5"] is not None
            else "n/a"
        )
        print(
            f"{i:4d}  {row['count']:4d}  {mean_p:>12}  {mean_rank:>9}  "
            f"{frac:>7}  {tag:<22}  "
            f"{row['true_cluster']!r} → {row['predicted_cluster']!r}"
        )
        top5_str = ", ".join(
            f"{t['cluster']!r}:{t['prob']:.3f}" for t in row["example_top5"]
        )
        print(f"       example top-5: [{top5_str}]")


def run_probe2(
    output_root: Path,
    data_root: Path,
    script: str,
    condition: str,
    seed: int,
    n_samples: int,
    out_path: Path,
    device_str: str = "cpu",
) -> None:
    """
    Run GT-aligned confusion extraction for one checkpoint.

    Prints the resolved checkpoint/tokenizer paths up front so a
    pre-#47 naming mismatch fails loudly with the expected filename.
    """
    device = torch.device(device_str)
    ckpt = checkpoint_path(str(output_root), script, condition, seed)
    tok = tokenizer_path(str(output_root), script, condition)
    print(f"[probe2] checkpoint: {ckpt}")
    print(f"[probe2] tokenizer:  {tok}")
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"missing checkpoint at {ckpt} "
            f"(expected script-scoped name per DECISIONS.md #47; "
            f"legacy checkpoint_{condition}_seed{seed}.pt is not used)"
        )

    model, tokenizer = load_model_and_tokenizer(
        output_root, script, condition, seed, device,
    )

    repo_root = resolve_repo_root(data_root)
    tasks = build_hindi_sample(data_root, repo_root, n_samples)
    completed = load_completed_paths(out_path)
    pending = [t for t in tasks if t["image_path"] not in completed]

    total = len(tasks)
    already = total - len(pending)
    print(
        f"[probe2] {script}/{condition}/seed={seed}: "
        f"{total} images ({already} done, {len(pending)} remaining)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_misreads: list[dict] = []

    # Reload prior misreads for end-of-run aggregation after resume.
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            all_misreads.extend(json.loads(line).get("misreads", []))

    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            row = task["row"]
            image = resize_to_canonical_height(Image.open(task["image_path"]))
            gen = run_generate(
                model, tokenizer, image, device, return_full_probs=True,
            )
            misreads = extract_misreads(gen, row.get("text") or "", tokenizer)
            record = {
                "checkpoint_script": script,
                "training_condition": condition,
                "seed": seed,
                "image_path": task["image_path"],
                "image_id": row.get("id"),
                "ground_truth": row.get("text"),
                "prediction": gen["text"],
                "n_gt_glyphs": len(grapheme_clusters(row.get("text") or "")),
                "n_pred_glyphs": len(
                    content_token_clusters(gen["token_ids"], tokenizer)
                ),
                "n_substitutions": len(misreads),
                "misreads": misreads,
                "checkpoint_path": ckpt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            all_misreads.extend(misreads)

            done = already + i + 1
            print(
                f"[probe2] {done}/{total} id={row.get('id')}  "
                f"subs={len(misreads)}  pred={gen['text'][:40]!r}"
            )

    pairs = aggregate_confusion_pairs(all_misreads)
    print_top_pairs(pairs, n=15)

    probs = [m["true_prob"] for m in all_misreads]
    ranks = [m["true_rank"] for m in all_misreads if m["true_rank"] is not None]
    in_top5 = [m["true_in_top5"] for m in all_misreads]
    print(
        f"\n[probe2] overall on {len(all_misreads)} substitutions: "
        f"mean_p(true)={np.mean(probs):.4f}  "
        f"mean_rank(true)={np.mean(ranks):.1f}  "
        f"frac_in_top5={np.mean(in_top5):.3f}"
        if all_misreads
        else "\n[probe2] no substitutions recorded"
    )
    print(f"[probe2] wrote {out_path} ({len(pairs)} unique confusion pairs)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe 2: GT-aligned confusion structure with true-token mass",
    )
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument(
        "--condition", default="natural", choices=["natural", "flattened", "inverted"],
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OCR_DATA_ROOT", "data"),
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Cap on Hindi images; same Random(0) draw as Probe 5b",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_probe2(
        Path(args.output_root),
        Path(args.data_root),
        args.script,
        args.condition,
        args.seed,
        args.n_samples,
        Path(args.out),
        args.device,
    )


if __name__ == "__main__":
    main()

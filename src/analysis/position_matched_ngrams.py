"""
Position-bucket n-gram references (Step 4a) from committed manifests + GT jsonl.

Why this exists: the paper compares instrument bucket means against
rest-of-sequence n-gram scores. Those references mix all positions ≥ 1.
This script scores the same add-α n-gram LMs only on tokens that fall
in each Follow-Up 6 bucket (0, 1, 2–9, 10–19, 20–39, 40+).

Step 4b (KL(model || 5-gram) and model-argmax = 5-gram-argmax) needs
the full position-t softmax. Committed GT-likelihood jsonl stores only
p(GT) and entropy, not the argmax or the full distribution. 4b is not
computed here.

Called from: `python src/analysis/position_matched_ngrams.py`
Output: `docs/position_matched_ngrams.md`
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import regex

_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from paper_defensibility_stats import load_jsonl  # noqa: E402

BOS, EOS = "<BOS>", "<EOS>"
BUCKETS = [
    ("Position 0", 0, 0),
    ("Position 1", 1, 1),
    ("Positions 2–9", 2, 9),
    ("Positions 10–19", 10, 19),
    ("Positions 20–39", 20, 39),
    ("Positions 40+", 40, 10**9),
]
ALPHA = 0.01


def graphemes(t: str) -> list[str]:
    return regex.findall(r"\X", t or "")


def build_ngram_counts(texts: list[str], n: int) -> tuple[Counter, Counter]:
    counts: Counter = Counter()
    context_counts: Counter = Counter()
    for t in texts:
        gs = [BOS] * (n - 1) + graphemes(t) + [EOS]
        for i in range(n - 1, len(gs)):
            ctx = tuple(gs[i - n + 1 : i]) if n > 1 else ()
            tok = gs[i]
            counts[ctx + (tok,)] += 1
            context_counts[ctx] += 1
    return counts, context_counts


def score_bucket(
    texts: list[str],
    counts: Counter,
    context_counts: Counter,
    n: int,
    v_est: int,
    lo: int,
    hi: int,
    alpha: float = ALPHA,
) -> tuple[float, int]:
    """
    Mean log p of GT graphemes whose index i is in [lo, hi], inclusive.

    i is the content-token index (0 = first grapheme), matching
    teacher-forced step_log_p_gt indices in the GT-likelihood jsonl.
    """
    logps: list[float] = []
    for t in texts:
        gs = graphemes(t)
        seq = [BOS] * (n - 1) + gs
        for i, tok in enumerate(gs):
            if i < lo or i > hi:
                continue
            ctx = tuple(seq[i : i + n - 1]) if n > 1 else ()
            c_ctx = context_counts[ctx]
            c_joint = counts[ctx + (tok,)]
            p = (c_joint + alpha) / (c_ctx + alpha * v_est)
            logps.append(math.log(p))
    if not logps:
        return float("nan"), 0
    return float(np.mean(logps)), len(logps)


def train_eval_split(repo: Path) -> tuple[list[str], list[str], int]:
    manifest = load_jsonl(repo / "data/manifests/hindi_natural.jsonl")
    gt_rows = [
        r
        for r in load_jsonl(
            repo / "data/probe_results/probe_gt_likelihood_hindi_natural_seed0.jsonl"
        )
        if r["condition"] == "real"
    ]
    eval_texts = [r["ground_truth"] for r in gt_rows]
    eval_set = set(eval_texts)
    train_texts = [r["text"] for r in manifest if r["text"] not in eval_set]
    all_g: set[str] = set()
    for t in train_texts:
        all_g.update(graphemes(t))
    v_est = len(all_g) + 2
    return train_texts, eval_texts, v_est


def main() -> None:
    repo = _ROOT
    train_texts, eval_texts, v_est = train_eval_split(repo)
    lines = [
        "# Position-matched n-gram references (Step 4a)",
        "",
        "**Source:** `data/manifests/hindi_natural.jsonl` excluding the 60 Probe",
        "GT-likelihood real strings (same split as",
        "`docs/paper_defensibility_stats.md` §7).",
        f"**Train lines:** {len(train_texts)}. **Eval strings:** {len(eval_texts)}.",
        f"**V_est:** {v_est} (unique train graphemes + 2). **add-α:** {ALPHA}.",
        "",
        "Instrument bucket means are recomputed from",
        "`data/probe_results/probe_gt_likelihood_hindi_natural_seed{0,1,2}.jsonl`",
        "(same pooling as Follow-Up 6).",
        "",
        "Step 4b (mean KL(model ‖ 5-gram) and argmax agreement) is **not computed**:",
        "committed `probe_gt_likelihood_*.jsonl` stores `step_p_gt` / entropy only,",
        "not the full softmax or the model argmax.",
        "",
        "| Bucket | n tokens (eval) | unigram | bigram | trigram | 4-gram | 5-gram | instrument real (FU6) | instrument blank (FU6) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fu6 = {}
    for cond in ("real", "blank"):
        by_pos = {name: [] for name, _, _ in BUCKETS}
        for s in range(3):
            path = repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
            for r in load_jsonl(path):
                if r["condition"] != cond:
                    continue
                for i, lp in enumerate(r["step_log_p_gt"]):
                    for name, lo, hi in BUCKETS:
                        if lo <= i <= hi:
                            by_pos[name].append(lp)
                            break
        fu6[cond] = {k: float(np.mean(v)) if v else float("nan") for k, v in by_pos.items()}
    models = {}
    for n in range(1, 6):
        models[n] = build_ngram_counts(train_texts, n)

    for name, lo, hi in BUCKETS:
        cells = []
        n_tok = None
        for n in range(1, 6):
            counts, ctx = models[n]
            mean_lp, n_t = score_bucket(
                eval_texts, counts, ctx, n, v_est, lo, hi
            )
            n_tok = n_t
            cells.append(f"{mean_lp:.4f}" if n_t else "—")
        ir, ib = fu6["real"][name], fu6["blank"][name]
        lines.append(
            f"| {name} | {n_tok} | "
            + " | ".join(cells)
            + f" | {ir:.4f} | {ib:.4f} |"
        )

    out = repo / "docs" / "position_matched_ngrams.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""
Step 4b: per-position KL(model || 5-gram) and argmax agreement.

Why this exists: committed GT-likelihood jsonl stores scalar p(GT) and
entropy, not the full softmax. Exclusivity vs a 5-gram prior needs
KL(p_model || p_5gram) and whether argmax(p_model) = argmax(p_5gram)
at each teacher-forced step, bucketed like Table 6 / Follow-Up 6.

This module:
  - maps an add-α 5-gram next-grapheme distribution onto the instrument
    tokenizer (same train/eval split as position_matched_ngrams.py);
  - aggregates compact probe jsonl written by src/probes/probe_ngram_kl.py.

It does not invent KL from p(GT) alone.

Called from: `python src/analysis/ngram_kl_argmax.py`
Output: `docs/ngram_kl_argmax.md`
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[2]
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from position_matched_ngrams import (  # noqa: E402
    ALPHA,
    BOS,
    BUCKETS,
    EOS,
    graphemes,
    load_jsonl,
)

PROBE = "data/probe_results/probe_ngram_kl_hindi_natural_seed{}.jsonl"


def ngram_next_dist(
    ctx: tuple,
    counts: Counter,
    context_counts: Counter,
    alphabet: set[str],
    v_est: int,
    alpha: float = ALPHA,
) -> dict[str, float]:
    """
    Add-α P(next | ctx) over `alphabet` (train graphemes ∪ {EOS}).

    v_est matches 4a (unique train graphemes + 2). Mass on symbols not
    in `alphabet` is left implicit in the +α V denominator.
    """
    c_ctx = context_counts[ctx]
    denom = c_ctx + alpha * v_est
    return {g: (counts[ctx + (g,)] + alpha) / denom for g in alphabet}


def scatter_onto_tokenizer(
    dist: dict[str, float],
    cluster_to_id: dict[str, int],
    vocab_size: int,
    rare_id: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Put 5-gram mass on tokenizer ids. Unknown graphemes go to <RARE>.
    PAD/BOS stay unused except for the floor+renormalize so q is a
    distribution on the same support as the decoder softmax.
    """
    q = np.zeros(vocab_size, dtype=np.float64)
    for g, p in dist.items():
        q[int(cluster_to_id.get(g, rare_id))] += float(p)
    q = np.maximum(q, eps)
    q /= q.sum()
    return q


def context_at_step(gt_text: str, step_i: int, n: int = 5) -> tuple:
    """
    5-gram context for teacher-forced step i.

    Content steps 0..L-1 predict grapheme i; step L predicts <EOS>.
    Matches generate() force_next_ids = encode(gt)[1:] (content + EOS).
    """
    gs = graphemes(gt_text)
    seq = [BOS] * (n - 1) + gs + [EOS]
    # next token is seq[(n-1) + step_i]
    pos = (n - 1) + step_i
    return tuple(seq[pos - (n - 1) : pos])


def kl_np(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.maximum(p.astype(np.float64), eps)
    q = np.maximum(q.astype(np.float64), eps)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def bucket_name(i: int) -> str | None:
    for name, lo, hi in BUCKETS:
        if lo <= i <= hi:
            return name
    return None


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.4f}"


def aggregate(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, dict[str, list]] = {
        name: {"kl": [], "agree": []} for name, _, _ in BUCKETS
    }
    for r in rows:
        if r.get("condition") not in (None, "real"):
            continue
        kls = r.get("step_kl_m_5gram") or []
        agrees = r.get("step_argmax_agree") or []
        for i, (kl, ag) in enumerate(zip(kls, agrees)):
            name = bucket_name(i)
            if name is None:
                continue
            if kl is not None:
                by[name]["kl"].append(float(kl))
            if ag is not None:
                by[name]["agree"].append(float(bool(ag)))
    out = {}
    for name, _, _ in BUCKETS:
        kls = by[name]["kl"]
        ags = by[name]["agree"]
        out[name] = {
            "n": len(kls),
            "mean_kl": float(np.mean(kls)) if kls else float("nan"),
            "agree_rate": float(np.mean(ags)) if ags else float("nan"),
        }
    return out


def write_doc(repo: Path) -> str:
    missing = []
    per_seed = []
    pooled_rows: list[dict] = []
    for s in range(3):
        path = repo / PROBE.format(s)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path.relative_to(repo)))
            continue
        rows = [r for r in load_jsonl(path) if r.get("condition", "real") == "real"]
        if not rows:
            missing.append(str(path.relative_to(repo)) + " (no real rows)")
            continue
        per_seed.append((s, aggregate(rows)))
        pooled_rows.extend(rows)

    lines = [
        "# n-gram KL and argmax agreement (Step 4b)",
        "",
        "Target: convert Section 8's consistency claim (mid-sequence",
        "teacher-forced log p(GT) ≈ 4-to-5-gram grapheme prior) into an",
        "exclusivity claim: is the **full** decoder softmax the 5-gram",
        "distribution, not merely matching it on the GT token?",
        "",
        "Committed `probe_gt_likelihood_hindi_natural_seed*.jsonl` cannot",
        "answer this: `score_one` in `src/probes/probe_gt_likelihood.py`",
        "requests `return_full_probs=True` then **discards** the tensor",
        "after entropy. No argmax, no KL(model ‖ 5-gram).",
        "",
        "Producer (needs checkpoints): `src/probes/probe_ngram_kl.py`.",
        "5-gram: add-α, α=0.01, same train/eval split as",
        "`src/analysis/position_matched_ngrams.py` (manifest minus the 60",
        "GT-likelihood real strings). KL uses generate.kl_divergence's",
        "floor (1e-8) after scattering the 5-gram onto the tokenizer.",
        "",
        "Buckets match Follow-Up 6 / Table 6: Position 0, 1, 2–9, 10–19,",
        "20–39, 40+ on teacher-forced step index (content + trailing EOS).",
        "",
    ]

    if missing and not pooled_rows:
        lines += [
            "**Status: not computed.** This laptop has no Hindi instrument",
            "checkpoints (`docs/tier0_checkpoint_status.md`). Compact jsonl",
            "was not written. Missing:",
        ]
        for m in missing:
            lines.append(f"- `{m}`")
        lines += [
            "",
            "Command on T4 or CPU once `--output-root` has",
            "`checkpoint_hindi_natural_seed{0,1,2}.pt` and",
            "`tokenizer_hindi_natural.json`:",
            "",
            "```bash",
            "for s in 0 1 2; do",
            "  PYTHONPATH=src python src/probes/probe_ngram_kl.py \\",
            "    --script hindi --condition natural --seed $s \\",
            "    --output-root checkpoints --data-root data --n-samples 100 \\",
            "    --device cuda \\",
            "    --out data/probe_results/probe_ngram_kl_hindi_natural_seed${s}.jsonl",
            "done",
            "PYTHONPATH=src python src/analysis/ngram_kl_argmax.py",
            "```",
            "",
            "Do not backfill KL from `step_p_gt`. That is one coordinate of",
            "the softmax, not the distribution.",
            "",
        ]
        text = "\n".join(lines)
        (repo / "docs" / "ngram_kl_argmax.md").write_text(text, encoding="utf-8")
        return text

    def table(title: str, stats: dict) -> list[str]:
        block = [
            f"### {title}",
            "",
            "| Bucket | n steps | mean KL(model ‖ 5-gram) | argmax agreement |",
            "|---|---:|---:|---:|",
        ]
        for name, _, _ in BUCKETS:
            st = stats[name]
            block.append(
                f"| {name} | {st['n']} | {fmt(st['mean_kl'])} | {fmt(st['agree_rate'])} |"
            )
        block.append("")
        return block

    for s, stats in per_seed:
        lines.extend(table(f"Seed {s}", stats))
    if pooled_rows and len(per_seed) > 1:
        seed_list = ",".join(str(s) for s, _ in per_seed)
        title = (
            "Pooled (seeds 0–2)"
            if len(per_seed) == 3
            else f"Pooled (seeds {seed_list} only)"
        )
        lines.extend(table(title, aggregate(pooled_rows)))
    if missing:
        lines += ["**Partial.** Missing or empty:"] + [f"- `{m}`" for m in missing] + [""]

    text = "\n".join(lines)
    (repo / "docs" / "ngram_kl_argmax.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    text = write_doc(_ROOT)
    print(text, end="")
    print(f"[ngram_kl] wrote {_ROOT / 'docs' / 'ngram_kl_argmax.md'}")


if __name__ == "__main__":
    main()

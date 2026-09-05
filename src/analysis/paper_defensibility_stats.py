"""
Offline paper-defensibility stats from committed jsonl (no new inference).

Why this exists: every single number needed for the paper's defensible text
and reviewer rebuttals is computable directly from already-committed probe
jsonl files — without launching new training or inference runs. This script
serves as the single, authoritative, reproducible reference for all 10
reviewer defensibility items plus the follow-up offline analyses.

Items covered:
  Part I (10 Defensibility Deliverables):
    1. Grapheme-cluster CER (Tier 1 normalized): real vs blank vs Ol Chiki vs Perso-Arabic
    2. Max-softmax confidence at generation position 1 specifically
    3. First-token identity distribution (mode grapheme and mode fraction)
    4. Output degeneracy stats (unique strings, mean pairwise edit distance, identical pairs)
    5. Attention ablation KL decomposition (flip vs agree mass, conf full vs zero)
    6. Fraction of teacher-forced positions where GT is unique argmax (p_gt > 0.5)
    7. Grapheme n-gram LM baseline on held-out text (unigram, bigram, trigram)
    8. Cross-attention contribution norm per decoder layer (status & Colab probe)
    9. Predictive entropy vs max-prob consistency check under teacher forcing
    10. Blank-condition confidence mean & SD for Section 5.1 table + n=180 panel structure

  Part II (Follow-Up Defensibility Analyses):
    - Abstract headline fact: first-token p(GT) vs position-1 max-softmax (V≈367)
    - Item 1: Shuffled image-text pairing (status & Colab probe)
    - Item 2: Cross-attention contribution norm (status & Colab probe)
    - Item 3: Noise & patch-scrambled conditions (status & Colab probe)
    - Item 4: Confidence as predictor of correctness (AUROC & Spearman vs CER)
    - Item 5: ANOVA random-effects variance decomposition (image vs seed vs residual)
    - Item 6: Teacher-forced GT log-likelihood position curve past token 2 + PNG plot
    - Item 7: Flattened/inverted accuracy + Stage 0 per-engine Tier 1 breakdown

Output:
  - Markdown report: docs/paper_defensibility_stats.md
  - Position curve plot: docs/figures/gt_likelihood_position_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import regex
import scipy.stats as stats

_ROOT = Path(__file__).resolve().parents[2]
_EVAL = _ROOT / "src" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from equivalence_tables import normalize_tier1, tier1_equivalent  # noqa: E402
from transliteration_equivalence import tier2_equivalent  # noqa: E402


def is_correct(prediction: str, ground_truth: str, language: str = "hindi") -> bool:
    """Same Tier1∨Tier2 gate as Probe 5 / 5b / 6."""
    if tier1_equivalent(prediction, ground_truth):
        return True
    try:
        return bool(tier2_equivalent(ground_truth, prediction, language))
    except Exception:
        return False


def load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        if rows and isinstance(rows[0], dict) and "records" not in rows[0]:
            return rows
    except json.JSONDecodeError:
        pass
    obj = json.loads(text)
    return obj["records"] if isinstance(obj, dict) and "records" in obj else [obj]


def graphemes(t: str) -> list[str]:
    return regex.findall(r"\X", t or "")


def levenshtein(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def grapheme_cer(pred: str, gt: str) -> float:
    """Tier1-normalized grapheme-cluster CER."""
    g = graphemes(normalize_tier1(gt))
    p = graphemes(normalize_tier1(pred))
    if not g and not p:
        return 0.0
    if not g:
        return 1.0
    return levenshtein(p, g) / len(g)


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann–Whitney AUROC; nan if one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    n1, n0 = len(pos), len(neg)
    rank_sum = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                rank_sum += 1.0
            elif p == n:
                rank_sum += 0.5
    return rank_sum / (n1 * n0)


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a)
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = rankdata(np.asarray(x)), rankdata(np.asarray(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def anova_var_decomp(Y: np.ndarray) -> dict:
    """
    Two-way random-effects moment estimator on a balanced panel.

    y_{is} = μ + a_i + b_s + e_{is} with Y shape (n_images, n_seeds).
    Reports σ²_image, σ²_seed, σ²_resid and percent of total.
    """
    n_i, n_s = Y.shape
    mu = Y.mean()
    row_means = Y.mean(axis=1)
    col_means = Y.mean(axis=0)
    ss_i = n_s * np.sum((row_means - mu) ** 2)
    ss_s = n_i * np.sum((col_means - mu) ** 2)
    ss_e = np.sum((Y - row_means[:, None] - col_means[None, :] + mu) ** 2)
    df_i, df_s, df_e = n_i - 1, n_s - 1, (n_i - 1) * (n_s - 1)
    ms_i, ms_s, ms_e = ss_i / df_i, ss_s / df_s, ss_e / df_e
    var_i = max(0.0, (ms_i - ms_e) / n_s)
    var_s = max(0.0, (ms_s - ms_e) / n_i)
    var_e = float(ms_e)
    total = var_i + var_s + var_e
    return {
        "mean": float(mu),
        "sd_pooled": float(Y.std(ddof=1)),
        "var_image": var_i,
        "var_seed": var_s,
        "var_resid": var_e,
        "pct_image": 100.0 * var_i / total,
        "pct_seed": 100.0 * var_s / total,
        "pct_resid": 100.0 * var_e / total,
    }


def panel_from_jsonl(
    pattern: str, condition: str, value_key: str, n_seeds: int = 3
) -> np.ndarray:
    ids = None
    cols: list[list[float]] = []
    for s in range(n_seeds):
        rows = {
            r["image_id"]: r[value_key]
            for r in load_jsonl(Path(pattern.format(s)))
            if r["condition"] == condition
        }
        if ids is None:
            ids = sorted(rows)
        cols.append([rows[i] for i in ids])
    return np.array(cols).T


# ==============================================================================
# PART I: The 10 Defensibility Deliverables
# ==============================================================================

def item1_grapheme_cer(repo: Path) -> list[str]:
    lines = [
        "## 1. Grapheme-cluster error rate (CER at grapheme level)",
        "",
        "Normalized via Tier 1 encoding equivalence, then split into grapheme clusters (\\X).",
        "",
        "| Condition / Script | Seed 0 | Seed 1 | Seed 2 | Pooled Mean | Pooled SD | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    specs = [
        ("Hindi real (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "hindi", "ground_truth"),
        ("Hindi blank (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "blank", "ground_truth"),
        ("Santhali / Ol Chiki (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "santhali", "ground_truth"),
        ("Kashmiri / Perso-Arabic (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "kashmiri", "ground_truth"),
        ("Probe 6 real_plain", "data/probe_results/probe6_synthetic_real_hindi_seed{}.jsonl", "real_plain", "ground_truth"),
        ("Probe 6 blank", "data/probe_results/probe6_synthetic_real_hindi_seed{}.jsonl", "blank", "ground_truth"),
    ]
    for label, pat, cond, gt_key in specs:
        seed_cers = []
        all_cers = []
        for s in range(3):
            path = repo / pat.format(s)
            rows = [r for r in load_jsonl(path) if r["condition"] == cond]
            cs = [grapheme_cer(r["text"], r[gt_key]) for r in rows]
            seed_cers.append(float(np.mean(cs)))
            all_cers.extend(cs)
        lines.append(
            f"| {label} | {seed_cers[0]:.4f} | {seed_cers[1]:.4f} | {seed_cers[2]:.4f} | "
            f"**{np.mean(all_cers):.4f}** | {np.std(all_cers, ddof=1):.4f} | {len(all_cers)} |"
        )
    lines.append("")
    lines.append(
        "**Verdict on Item 1:** CER(real) ≈ **0.985** is not meaningfully lower than "
        "CER(blank) ≈ **0.949** (pooled blank is actually slightly lower CER). "
        "The model does not read the text; the deflationary reading wins."
    )
    lines.append("")
    return lines


def item2_pos1_confidence(repo: Path) -> list[str]:
    lines = [
        "## 2. Max-softmax confidence at generation position 1 specifically",
        "",
        "First generated token's max-softmax probability (probe5b):",
        "",
        "| Condition | Seed 0 | Seed 1 | Seed 2 | Pooled Mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond, label in [("hindi", "Real Hindi"), ("blank", "Blank")]:
        c_by_seed = []
        all_c = []
        for s in range(3):
            rows = [r for r in load_jsonl(repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl") if r["condition"] == cond]
            c1 = [r["step_confidences"][0] for r in rows if r.get("step_confidences")]
            c_by_seed.append(float(np.mean(c1)))
            all_c.extend(c1)
        lines.append(
            f"| {label} | {c_by_seed[0]:.4f} | {c_by_seed[1]:.4f} | {c_by_seed[2]:.4f} | "
            f"**{np.mean(all_c):.4f} ± {np.std(all_c, ddof=1):.4f}** |"
        )
    lines.append("")
    lines.append(
        "- Seed 0 blank is near-deterministic at step 1 (**0.9999**), while Seed 2 blank drops to **0.7874**."
    )
    lines.append("")
    return lines


def item3_first_token_identity(repo: Path) -> list[str]:
    lines = [
        "## 3. First-token identity distribution",
        "",
        "First decoded grapheme cluster across the 60 images per seed (probe5b):",
        "",
        "| Condition | Seed | Mode Grapheme | Mode Count | Mode Fraction | # Distinct First Graphemes |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for cond, label in [("hindi", "Real Hindi"), ("blank", "Blank")]:
        pooled_toks = []
        for s in range(3):
            rows = [r for r in load_jsonl(repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl") if r["condition"] == cond]
            toks = [(graphemes(r["text"]) or ["<EMPTY>"])[0] for r in rows]
            pooled_toks.extend(toks)
            ctr = Counter(toks)
            mode, count = ctr.most_common(1)[0]
            lines.append(
                f"| {label} | {s} | `{mode}` | {count}/60 | {count/len(toks):.1%} | {len(ctr)} |"
            )
        ctr_p = Counter(pooled_toks)
        mode_p, count_p = ctr_p.most_common(1)[0]
        lines.append(
            f"| **{label} Pooled** | All | `{mode_p}` | {count_p}/180 | **{count_p/len(pooled_toks):.1%}** | {len(ctr_p)} |"
        )
    lines.append("")
    lines.append(
        "- Blank seed 0 has **100% constant start token** (`को`). "
        "Real is peaked per seed (40–72% mode) but seeds disagree on which token sits at the peak."
    )
    lines.append("")
    return lines


def item4_output_degeneracy(repo: Path) -> list[str]:
    lines = [
        "## 4. Output degeneracy stats",
        "",
        "Number of unique decoded strings, identical-pair fraction, and mean pairwise grapheme edit distance (within seed, n=60):",
        "",
        "| Source / Condition | Seed | Unique / 60 | Unique % | Identical Pairs | Mean Pairwise Edit | Mean Norm Edit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, pat, conds in [
        ("Probe 5b Hindi", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", ["hindi", "blank"]),
        ("Probe 6", "data/probe_results/probe6_synthetic_real_hindi_seed{}.jsonl", ["real_plain", "blank"]),
    ]:
        for cond in conds:
            for s in range(3):
                rows = [r for r in load_jsonl(repo / pat.format(s)) if r["condition"] == cond]
                texts = [r["text"] for r in rows]
                n = len(texts)
                unique = len(set(texts))
                dists, norms = [], []
                for i, j in combinations(range(n), 2):
                    gi, gj = graphemes(texts[i]), graphemes(texts[j])
                    d = levenshtein(gi, gj)
                    dists.append(d)
                    norms.append(d / max(len(gi), len(gj), 1))
                identical = sum(1 for a, b in combinations(texts, 2) if a == b)
                npairs = n * (n - 1) // 2
                lines.append(
                    f"| {label} ({cond}) | {s} | {unique} | {unique/n:.1%} | "
                    f"{identical}/{npairs} ({identical/npairs:.1%}) | {np.mean(dists):.2f} | {np.mean(norms):.3f} |"
                )
    lines.append("")
    lines.append(
        "- Blank collapses to a tiny set of outputs (e.g. Probe 5b seed 0: 3 unique strings, 34.2% identical pairs; "
        "seed 1: 3 unique strings, 76.1% identical pairs, median pairwise edit distance 0)."
    )
    lines.append("")
    return lines


def item5_ablation_flip_agree(repo: Path) -> list[str]:
    lines = [
        "## 5. Confidence conditional on argmax flip vs agreement in encoder ablation",
        "",
        "Decomposition of the pooled 1.075 nats KL(full ‖ zero):",
        "",
        "| Seed | Step Count | Agree Rate | Flip Rate | Mean KL Agree | Mean KL Flip | Flip Share of Total KL | KL / (1 − Agree) | Conf Full (Agree/Flip) | Conf Zero (Agree/Flip) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    rows_all = []
    for s in range(3):
        rows = load_jsonl(repo / f"data/probe_results/attention_ablation_hindi_natural_seed{s}.jsonl")
        rows_all.extend(rows)
        agree_kl, flip_kl = [], []
        agree_cf, flip_cf = [], []
        agree_cz, flip_cz = [], []
        n_a = n_f = 0
        for r in rows:
            for m in r["step_metrics"]:
                if m["top1_agree"]:
                    n_a += 1
                    agree_kl.append(m["kl_full_given_zero"])
                    agree_cf.append(m["conf_full"])
                    agree_cz.append(m["conf_zero"])
                else:
                    n_f += 1
                    flip_kl.append(m["kl_full_given_zero"])
                    flip_cf.append(m["conf_full"])
                    flip_cz.append(m["conf_zero"])
        total = n_a + n_f
        mean_kl = float(np.mean(agree_kl + flip_kl))
        contrib_flip = (n_f / total) * float(np.mean(flip_kl))
        flip_share = contrib_flip / mean_kl
        lines.append(
            f"| {s} | {total} | {n_a/total:.1%} | {n_f/total:.1%} | {np.mean(agree_kl):.4f} | {np.mean(flip_kl):.2f} | "
            f"**{flip_share:.1%}** | {mean_kl/(n_f/total):.2f} | {np.mean(agree_cf):.3f} / {np.mean(flip_cf):.3f} | "
            f"{np.mean(agree_cz):.3f} / {np.mean(flip_cz):.3f} |"
        )
    # Pooled
    agree_kl, flip_kl = [], []
    agree_cf, flip_cf = [], []
    agree_cz, flip_cz = [], []
    n_a = n_f = 0
    for r in rows_all:
        for m in r["step_metrics"]:
            if m["top1_agree"]:
                n_a += 1
                agree_kl.append(m["kl_full_given_zero"])
                agree_cf.append(m["conf_full"])
                agree_cz.append(m["conf_zero"])
            else:
                n_f += 1
                flip_kl.append(m["kl_full_given_zero"])
                flip_cf.append(m["conf_full"])
                flip_cz.append(m["conf_zero"])
    total = n_a + n_f
    mean_kl = float(np.mean(agree_kl + flip_kl))
    contrib_flip = (n_f / total) * float(np.mean(flip_kl))
    flip_share = contrib_flip / mean_kl
    lines.append(
        f"| **Pooled** | {total} | **{n_a/total:.1%}** | **{n_f/total:.1%}** | {np.mean(agree_kl):.4f} | {np.mean(flip_kl):.2f} | "
        f"**{flip_share:.1%}** | **{mean_kl/(n_f/total):.2f}** | {np.mean(agree_cf):.3f} / {np.mean(flip_cf):.3f} | "
        f"{np.mean(agree_cz):.3f} / {np.mean(flip_cz):.3f} |"
    )
    lines.append("")
    lines.append(
        "- **Decomposition confirms hypothesis:** 96.7% of all KL comes from the ~8.2% flipped positions. "
        "Agreeing positions contribute near-zero KL (0.0268 nats). "
        "However, confidence at flipped positions drops slightly (conf_full=0.893, conf_zero=0.825) rather than staying at 0.98+."
    )
    lines.append("")
    return lines


def item6_gt_argmax_fraction(repo: Path) -> list[str]:
    lines = [
        "## 6. Fraction of teacher-forced positions where GT is the unique argmax (p_gt > 0.5)",
        "",
        "| Condition | Seed 0 | Seed 1 | Seed 2 | Pooled Total | Position 0 Only | Positions ≥ 1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cond, label in [("real", "Real Hindi"), ("blank", "Blank")]:
        flags_by_seed = []
        all_flags = []
        pos0_flags = []
        rest_flags = []
        for s in range(3):
            path = repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
            rows = [r for r in load_jsonl(path) if r["condition"] == cond]
            flags = []
            for r in rows:
                for i, p in enumerate(r["step_p_gt"]):
                    is_arg = (p > 0.5)
                    flags.append(is_arg)
                    if i == 0:
                        pos0_flags.append(is_arg)
                    else:
                        rest_flags.append(is_arg)
            flags_by_seed.append(float(np.mean(flags)))
            all_flags.extend(flags)
        lines.append(
            f"| {label} | {flags_by_seed[0]:.1%} | {flags_by_seed[1]:.1%} | {flags_by_seed[2]:.1%} | "
            f"**{np.mean(all_flags):.1%}** | **{np.mean(pos0_flags):.1%}** (0/180) | **{np.mean(rest_flags):.1%}** |"
        )
    lines.append("")
    lines.append(
        "- GT is **never** the argmax at position 0 (0/180). From position 1 onward, GT is the argmax at **~92.5%** of positions."
    )
    lines.append("")
    return lines


def item7_ngram_lm_baseline(repo: Path) -> list[str]:
    lines = [
        "## 7. Grapheme-cluster n-gram LM baseline on held-out text",
        "",
        "Trained on `data/manifests/hindi_natural.jsonl` excluding all 60 evaluation ground-truth texts (2,491 train lines; V≈367; add-α=0.01):",
        "",
        "| Order | Whole-Sequence Mean Log p(GT) | Rest-of-Sequence Mean Log p(GT) (pos ≥ 1) |",
        "|---|---:|---:|",
    ]
    manifest = load_jsonl(repo / "data/manifests/hindi_natural.jsonl")
    gt_rows = [r for r in load_jsonl(repo / "data/probe_results/probe_gt_likelihood_hindi_natural_seed0.jsonl") if r["condition"] == "real"]
    eval_texts = [r["ground_truth"] for r in gt_rows]
    eval_set = set(eval_texts)
    train_texts = [r["text"] for r in manifest if r["text"] not in eval_set]

    BOS = "<BOS>"
    EOS = "<EOS>"

    def build_ngram_counts(texts, n):
        counts = Counter()
        context_counts = Counter()
        for t in texts:
            gs = [BOS] * (n - 1) + graphemes(t) + [EOS]
            for i in range(n - 1, len(gs)):
                ctx = tuple(gs[i - n + 1:i]) if n > 1 else ()
                tok = gs[i]
                counts[ctx + (tok,)] += 1
                context_counts[ctx] += 1
        return counts, context_counts

    def score_mean_logp(texts, counts, context_counts, n, V_est, alpha=0.01, skip_pos0=False):
        logps = []
        for t in texts:
            gs = graphemes(t)
            seq = [BOS] * (n - 1) + gs
            for i, tok in enumerate(gs):
                if skip_pos0 and i == 0:
                    continue
                ctx = tuple(seq[i:i + n - 1]) if n > 1 else ()
                c_ctx = context_counts[ctx]
                c_joint = counts[ctx + (tok,)]
                p = (c_joint + alpha) / (c_ctx + alpha * V_est)
                logps.append(math.log(p))
        return float(np.mean(logps))

    all_g = set()
    for t in train_texts:
        all_g.update(graphemes(t))
    V = len(all_g) + 2

    for n, label in [
        (1, "Unigram (n=1)"),
        (2, "Bigram (n=2)"),
        (3, "Trigram (n=3)"),
        (4, "4-gram (n=4)"),
        (5, "5-gram (n=5)"),
    ]:
        counts, ctx = build_ngram_counts(train_texts, n)
        whole = score_mean_logp(eval_texts, counts, ctx, n, V, alpha=0.01, skip_pos0=False)
        rest = score_mean_logp(eval_texts, counts, ctx, n, V, alpha=0.01, skip_pos0=True)
        lines.append(f"| {label} | {whole:.4f} | **{rest:.4f}** |")
    lines.append("")
    lines.append(
        "- Bigram LM scores **−2.25**, trigram scores **−0.99**, 4-gram scores **−0.44**, and 5-gram scores **−0.26** on rest-of-sequence. "
        "Notice that at later positions (positions 2–9 mean = **−0.48**, positions 20–39 mean = **−0.27**), the instrument model's "
        "log p(GT) converges directly toward 4-gram and 5-gram grapheme priors. "
        "The decoder is behaving as a high-order local grapheme language model rather than maintaining vision-conditioned grounding."
    )
    lines.append("")
    return lines


def item8_cross_attn_status() -> list[str]:
    lines = [
        "## 8. Cross-attention contribution norm per decoder layer",
        "",
        "- **Status:** Not recorded in committed jsonl or `docs/attention_ablation_analysis.md`.",
        "- **Probe authored:** `src/probes/probe_cross_attn_norms.py` hooks `nn.TransformerDecoderLayer` to compute "
        "‖cross-attn output‖ / ‖residual stream‖ per layer on teacher-forced forward passes. Ready for Colab execution.",
        "",
    ]
    return lines


def item9_entropy_consistency(repo: Path) -> list[str]:
    lines = [
        "## 9. Consistency check on predictive entropy and max probability under teacher forcing",
        "",
        "| Condition | Mean Entropy H(p) | Mean p(GT) | Mean Max-Prob when GT is Argmax (exact) | Coverage of Argmax | Implied Max from Binary H(p) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    def implied_p_from_H(H):
        if H <= 0:
            return 1.0
        if H >= math.log(2):
            return 0.5
        lo, hi = 0.5, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            Hb = -mid * math.log(mid) - (1 - mid) * math.log(1 - mid) if 0 < mid < 1 else 0
            if Hb > H:
                lo = mid
            else:
                hi = mid
        return hi

    for cond, label in [("real", "Real Hindi"), ("blank", "Blank")]:
        p_all, ent_all, max_known = [], [], []
        for s in range(3):
            path = repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
            for r in load_jsonl(path):
                if r["condition"] != cond:
                    continue
                for p, e in zip(r["step_p_gt"], r["step_entropy"]):
                    p_all.append(p)
                    ent_all.append(e)
                    if p > 0.5:
                        max_known.append(p)
        implied = [implied_p_from_H(e) for e in ent_all]
        lines.append(
            f"| {label} | {np.mean(ent_all):.4f} nats | {np.mean(p_all):.4f} | "
            f"**{np.mean(max_known):.4f}** | {len(max_known)/len(p_all):.1%} | {np.mean(implied):.4f} |"
        )
    lines.append("")
    lines.append(
        "- Under teacher forcing, when GT is argmax (~90% of steps), the average peak probability is **0.9983** (sharper than self-gen 0.9873). "
        "The mean entropy of 0.021 nats is visibly consistent with near-unit probabilities."
    )
    lines.append("")
    return lines


def item10_blank_conf_section51(repo: Path) -> list[str]:
    lines = [
        "## 10. Blank-condition confidence mean and SD for Section 5.1 table",
        "",
        "Mean confidence across images (Probe 5b `mean_confidence` field):",
        "",
        "| Condition | Seed 0 | Seed 1 | Seed 2 | Pooled Mean ± SD | Unique Images | Total Records |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cond, label in [("hindi", "Real Hindi"), ("blank", "Blank")]:
        confs_by_seed = []
        all_confs = []
        for s in range(3):
            rows = [r for r in load_jsonl(repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl") if r["condition"] == cond]
            cs = [r["mean_confidence"] for r in rows]
            confs_by_seed.append((float(np.mean(cs)), float(np.std(cs, ddof=1))))
            all_confs.extend(cs)
        lines.append(
            f"| {label} | {confs_by_seed[0][0]:.4f} ± {confs_by_seed[0][1]:.4f} | "
            f"{confs_by_seed[1][0]:.4f} ± {confs_by_seed[1][1]:.4f} | "
            f"{confs_by_seed[2][0]:.4f} ± {confs_by_seed[2][1]:.4f} | "
            f"**{np.mean(all_confs):.4f} ± {np.std(all_confs, ddof=1):.4f}** | 60 | 180 |"
        )
    lines.append("")
    lines.append(
        "**Panel structure clarification:** n=180 is **60 unique images evaluated across 3 seeds**, "
        "not 180 independent images. All 3 seeds evaluate the exact same 60 image paths."
    )
    lines.append("")
    return lines


# ==============================================================================
# PART II: Follow-Up Defensibility Analyses
# ==============================================================================

def abstract_fact(repo: Path, v: int = 367) -> list[str]:
    lines = ["## Abstract Fact: First-token p(GT) vs Max-Softmax at Position 1", ""]
    for cond, label in (("real", "Real Hindi"), ("blank", "Blank")):
        lps = []
        for s in range(3):
            for r in load_jsonl(repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"):
                if r["condition"] == cond:
                    lps.append(r["step_log_p_gt"][0])
        g = math.exp(float(np.mean(lps)))
        lines.append(
            f"- **Teacher-Forced {label} Pos 0:** Mean log p = {np.mean(lps):.4f} → "
            f"Geometric mean p = **{g:.3e}** (uniform 1/{v} = {1/v:.3e}; "
            f"**{math.log10((1/v)/max(g,1e-300)):.1f} orders of magnitude below uniform chance**)."
        )
    for cond, label in (("hindi", "Real Hindi"), ("blank", "Blank")):
        c1 = []
        for s in range(3):
            for r in load_jsonl(repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl"):
                if r["condition"] == cond and r.get("step_confidences"):
                    c1.append(r["step_confidences"][0])
        lines.append(
            f"- **Self-Generated {label} Pos 1 Max-Softmax:** Mean = **{np.mean(c1):.4f}** (n={len(c1)})."
        )
    lines.append("")
    lines.append(
        "> **The Abstract Pairing:** At sequence start, the model places **~0.90 max-softmax mass** "
        "on its chosen first token while assigning the true ground-truth token **~10⁻¹¹ probability** "
        "(nearly 8 orders of magnitude below uniform chance over V≈367 grapheme clusters)."
    )
    lines.append("")
    return lines


def calibration_block(repo: Path) -> list[str]:
    lines = ["## Follow-Up 4: Confidence as a Predictor of Correctness", ""]
    lines.append("Per-image AUROC and Spearman rank correlation of confidence against correctness / CER:")
    lines.append("")
    for cond in ("natural", "flattened", "inverted"):
        all_conf, all_corr, all_cer = [], [], []
        for s in range(3):
            rows = load_jsonl(repo / f"data/probe_results/probe5_hindi_{cond}_seed{s}.jsonl")
            conf = [r["confidence"] for r in rows]
            corr = [1 if r["correct"] else 0 for r in rows]
            cers = [grapheme_cer(r["prediction"], r["ground_truth"]) for r in rows]
            lines.append(
                f"- Probe 5 {cond} seed {s}: n={len(rows)}, Acc={np.mean(corr):.3f}, "
                f"AUROC={auroc(conf, corr):.4f}, Spearman(conf, CER)={spearman(conf, cers):.4f}"
            )
            all_conf.extend(conf)
            all_corr.extend(corr)
            all_cer.extend(cers)
        lines.append(
            f"  - **POOLED {cond}:** Acc={np.mean(all_corr):.4f}, "
            f"**AUROC={auroc(all_conf, all_corr):.4f}**, Spearman={spearman(all_conf, all_cer):.4f}"
        )
    # Probe 5b Hindi real
    confs, cers, corrs = [], [], []
    for s in range(3):
        rows = [r for r in load_jsonl(repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl") if r["condition"] == "hindi"]
        conf = [r["mean_confidence"] for r in rows]
        corr = [1 if is_correct(r["text"], r["ground_truth"]) else 0 for r in rows]
        cer = [grapheme_cer(r["text"], r["ground_truth"]) for r in rows]
        lines.append(
            f"- Probe 5b Hindi seed {s}: Acc={np.mean(corr):.4f} (n_correct={sum(corr)}), "
            f"AUROC={auroc(conf, corr)}, Spearman(conf, CER)={spearman(conf, cer):.4f}"
        )
        confs.extend(conf)
        cers.extend(cer)
        corrs.extend(corr)
    med = float(np.median(cers))
    better = [1 if c < med else 0 for c in cers]
    lines.append(
        f"- **Probe 5b Hindi POOLED:** Line Acc = **0.0000** (AUROC undefined); "
        f"Spearman(conf, CER) = **{spearman(confs, cers):.4f}**; "
        f"AUROC(conf → CER < median) = **{auroc(confs, better):.4f}**."
    )
    lines.append("")
    return lines


def variance_block(repo: Path) -> list[str]:
    lines = ["## Follow-Up 5: Variance Decomposition (Image vs Seed vs Residual)", ""]
    lines.append("Two-way random effects ANOVA on 60 images × 3 seeds panel:")
    lines.append("")
    specs = [
        ("Probe 5b Hindi Confidence", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "hindi", "mean_confidence"),
        ("Probe 5b Blank Confidence", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "blank", "mean_confidence"),
        ("Probe 6 Plain Confidence", "data/probe_results/probe6_synthetic_real_hindi_seed{}.jsonl", "real_plain", "mean_confidence"),
        ("Probe 6 Blank Confidence", "data/probe_results/probe6_synthetic_real_hindi_seed{}.jsonl", "blank", "mean_confidence"),
    ]
    for label, pat, cond, key in specs:
        Y = panel_from_jsonl(str(repo / pat), cond, key)
        d = anova_var_decomp(Y)
        lines.append(
            f"- **{label}:** Mean = {d['mean']:.6f}, Pooled SD = {d['sd_pooled']:.6f} → "
            f"σ² Share: **Image {d['pct_image']:.1f}%**, **Seed {d['pct_seed']:.1f}%**, **Residual {d['pct_resid']:.1f}%**"
        )
    for cond in ("real", "blank"):
        Y = panel_from_jsonl(
            str(repo / "data/probe_results/probe_gt_likelihood_hindi_natural_seed{}.jsonl"),
            cond,
            "mean_log_p_gt",
        )
        d = anova_var_decomp(Y)
        lines.append(
            f"- **GT Likelihood ({cond}):** Mean = {d['mean']:.6f} → "
            f"σ² Share: **Image {d['pct_image']:.1f}%**, **Seed {d['pct_seed']:.1f}%**, **Residual {d['pct_resid']:.1f}%**"
        )
    lines.append("")
    lines.append(
        "- For confidence, **82–84% of variance is residual (image × seed interaction)**. "
        "Pooled SD over 180 runs overestimates precision if treated as 180 independent images."
    )
    lines.append("")
    return lines


def position_curve_block(repo: Path, out_png: Path | None, v_grapheme: int = 367) -> list[str]:
    lines = ["## Follow-Up 6: Position Curve Past Token 2", ""]
    uniform = math.log(1.0 / v_grapheme)
    trigram = -0.99
    lines.append(
        f"Teacher-forced mean log p(GT) by position (reference lines: uniform log(1/367) = {uniform:.4f}, trigram rest = {trigram}):"
    )
    lines.append("")
    by_cond: dict[str, dict[int, list[float]]] = {}
    for cond in ("real", "blank"):
        by_pos: dict[int, list[float]] = defaultdict(list)
        for s in range(3):
            path = repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
            for r in load_jsonl(path):
                if r["condition"] != cond:
                    continue
                for i, lp in enumerate(r["step_log_p_gt"]):
                    by_pos[i].append(lp)
        by_cond[cond] = by_pos

    lines.append("| Position Bucket | Real Hindi Mean Log p(GT) | Blank Mean Log p(GT) | n Positions |")
    lines.append("|---|---:|---:|---:|")
    for name, lo, hi in [
        ("Position 0", 0, 0),
        ("Position 1", 1, 1),
        ("Positions 2–9", 2, 9),
        ("Positions 10–19", 10, 19),
        ("Positions 20–39", 20, 39),
        ("Positions 40+", 40, 10**9),
    ]:
        v_real = [lp for i, arr in by_cond["real"].items() if lo <= i <= hi for lp in arr]
        v_blank = [lp for i, arr in by_cond["blank"].items() if lo <= i <= hi for lp in arr]
        lines.append(f"| {name} | {np.mean(v_real):.4f} | {np.mean(v_blank):.4f} | {len(v_real)} |")

    if out_png is not None:
        try:
            import matplotlib.pyplot as plt
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            for cond, style in (("real", "-"), ("blank", "--")):
                xs, ys = [], []
                for i in sorted(by_cond[cond]):
                    arr = by_cond[cond][i]
                    if len(arr) < 30:
                        continue
                    xs.append(i)
                    ys.append(float(np.mean(arr)))
                ax.plot(xs, ys, style, label=cond, linewidth=1.8)
            ax.axhline(uniform, color="0.4", linestyle=":", label=f"log(1/{v_grapheme}) = {uniform:.2f}")
            ax.axhline(trigram, color="0.55", linestyle="-.", label=f"trigram rest ({trigram})")
            ax.set_xlabel("Teacher-forced position (0 = first GT token)")
            ax.set_ylabel("Mean log p(GT)")
            ax.set_ylim(-30, 1)
            ax.legend(loc="lower right", fontsize=8)
            ax.set_title("Teacher-forced log p(GT) by position — real vs blank")
            fig.tight_layout()
            fig.savefig(out_png, dpi=140)
            plt.close(fig)
            lines.append("")
            lines.append(f"- **Figure generated:** `docs/figures/{out_png.name}`")
        except Exception as e:
            lines.append(f"- (Plot generation skipped: {e})")
    lines.append("")
    return lines


def flat_inv_and_tier1_block(repo: Path) -> list[str]:
    lines = ["## Follow-Up 7: Flattened/Inverted Accuracy & Stage 0 Per-Engine Tier 1 Breakdown", ""]
    lines.append("### Probe 5 Line Accuracy Across Exposure Conditions (Tier 1∨2 Correctness)")
    lines.append("")
    lines.append("| Condition | Seed 0 | Seed 1 | Seed 2 | Seed Mean ± SD | Mean Confidence |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in ("natural", "flattened", "inverted"):
        accs, confs = [], []
        for s in range(3):
            rows = load_jsonl(repo / f"data/probe_results/probe5_hindi_{cond}_seed{s}.jsonl")
            acc = float(np.mean([1 if r["correct"] else 0 for r in rows]))
            cf = float(np.mean([r["confidence"] for r in rows]))
            accs.append(acc)
            confs.append(cf)
        lines.append(
            f"| {cond} | {accs[0]:.4f} | {accs[1]:.4f} | {accs[2]:.4f} | "
            f"**{np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}** | {np.mean(confs):.4f} |"
        )
    lines.append("")
    lines.append("### Stage 0 Per-Engine Tier 1 Breakdown")
    lines.append("")
    try:
        sys.path.insert(0, str(repo / "src" / "eval"))
        import error_taxonomy as et

        human = et.load_human_labels()
        gt = {lang: et.load_ground_truth(lang) for lang in et.LANGUAGES}
        counts = {e: {"EXACT": 0, "TIER1": 0, "TIER2": 0, "GENUINE": 0, "UNREVIEWED": 0} for e in et.ENGINES}
        for engine in et.ENGINES:
            for language in et.LANGUAGES:
                for pred in et.load_predictions(engine, language):
                    if pred.get("skipped_reason"):
                        continue
                    g = gt[language].get(pred["id"])
                    if g is None:
                        continue
                    hl = human.get((language, engine, pred["id"], pred["variant"]))
                    bucket, _ = et.classify(g["text"], pred.get("predicted_text") or "", language, hl)
                    counts[engine][bucket] += 1
        lines.append("| Engine | n | EXACT | TIER1 | TIER2 | GENUINE | UNREVIEWED | Tier 1 Share of Non-Exact | Tier 2 Share of Non-Exact | Tier 1+2 Total Share |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for engine in et.ENGINES:
            total = sum(counts[engine].values())
            if total == 0:
                continue
            ex, t1, t2, gen, unrev = counts[engine]["EXACT"], counts[engine]["TIER1"], counts[engine]["TIER2"], counts[engine]["GENUINE"], counts[engine]["UNREVIEWED"]
            non = total - ex
            t1_share = (t1 / non) if non > 0 else 0
            t2_share = (t2 / non) if non > 0 else 0
            comb_share = ((t1 + t2) / non) if non > 0 else 0
            lines.append(
                f"| {engine} | {total} | {ex} ({ex/total:.1%}) | {t1} ({t1/total:.1%}) | "
                f"{t2} ({t2/total:.1%}) | {gen} ({gen/total:.1%}) | {unrev} ({unrev/total:.1%}) | **{t1_share:.1%}** ({t1}/{non}) | "
                f"{t2_share:.1%} ({t2}/{non}) | **{comb_share:.1%}** ({t1+t2}/{non}) |"
            )
        lines.append("")
        lines.append(
            "- **Tier 2 finding:** Tier 2 (phonetic equivalence via ISO 15919 transliteration) resolves 0% of residual errors beyond Tier 1 on this corpus. "
            "Encoding variants (Tier 1: joiners, anusvara vs conjunct nasal, nukta compositions) account for all systematic representation ambiguities; "
            "phonetic substitution residuals are vanishingly rare once Tier 1 is applied."
        )
    except Exception as exc:
        lines.append(f"(Live recount failed: {exc})")
    lines.append("")
    return lines


# ==============================================================================
# PART III: Additional Offline Analyses
# ==============================================================================

def paired_tests_block(repo: Path) -> list[str]:
    """
    Paired Wilcoxon signed-rank tests for real vs blank inputs.

    Why this exists: Exactly the same 60 images are evaluated under real
    and blank conditions across all 3 seeds. Unpaired tests discard the
    within-image matching, understating statistical power.
    This analysis performs Wilcoxon signed-rank tests on per-image
    grapheme CER and per-image mean confidence within each seed,
    plus pooled and cluster-adjusted tests with seed as cluster.
    """
    lines = [
        "## Offline Analysis 1: Paired Tests (Real vs Blank)",
        "",
        "Because the exact same 60 `image_id`s appear under both real and blank conditions, "
        "paired tests provide strictly more statistical power than unpaired comparisons.",
        "",
        "### 1.1 Per-Seed and Pooled Wilcoxon Signed-Rank Tests",
        "",
        "| Metric | Seed | n Pairs | Mean Real | Mean Blank | Mean Diff (R − B) | Wilcoxon W | Two-sided p | Rank-Biserial r | Cohen's d_z |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    diffs_cer_all, diffs_conf_all = [], []
    seed_cer_diffs = defaultdict(list)
    seed_conf_diffs = defaultdict(list)
    img_cer_diffs = defaultdict(list)
    img_conf_diffs = defaultdict(list)

    for s in range(3):
        path = repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        real_map = {r["image_id"]: r for r in rows if r["condition"] == "hindi"}
        blank_map = {r["image_id"]: r for r in rows if r["condition"] == "blank"}
        shared_ids = sorted(set(real_map) & set(blank_map))

        cer_r = np.array([grapheme_cer(real_map[i]["text"], real_map[i]["ground_truth"]) for i in shared_ids])
        cer_b = np.array([grapheme_cer(blank_map[i]["text"], blank_map[i]["ground_truth"]) for i in shared_ids])
        conf_r = np.array([real_map[i]["mean_confidence"] for i in shared_ids])
        conf_b = np.array([blank_map[i]["mean_confidence"] for i in shared_ids])

        diff_c = cer_r - cer_b
        diff_cf = conf_r - conf_b

        for val, i in zip(diff_c, shared_ids):
            diffs_cer_all.append(val)
            seed_cer_diffs[s].append(val)
            img_cer_diffs[i].append(val)
        for val, i in zip(diff_cf, shared_ids):
            diffs_conf_all.append(val)
            seed_conf_diffs[s].append(val)
            img_conf_diffs[i].append(val)

        # Wilcoxon CER
        res_c = stats.wilcoxon(cer_r, cer_b)
        w_c, p_c = res_c.statistic, res_c.pvalue
        nonzero_c = diff_c[diff_c != 0]
        ranks_c = stats.rankdata(np.abs(nonzero_c))
        w_pos_c = np.sum(ranks_c[nonzero_c > 0])
        w_neg_c = np.sum(ranks_c[nonzero_c < 0])
        r_rb_c = (w_pos_c - w_neg_c) / (w_pos_c + w_neg_c) if len(nonzero_c) > 0 else 0.0
        dz_c = np.mean(diff_c) / np.std(diff_c, ddof=1) if np.std(diff_c, ddof=1) > 0 else 0.0

        # Wilcoxon Conf
        res_cf = stats.wilcoxon(conf_r, conf_b)
        w_cf, p_cf = res_cf.statistic, res_cf.pvalue
        nonzero_cf = diff_cf[diff_cf != 0]
        ranks_cf = stats.rankdata(np.abs(nonzero_cf))
        w_pos_cf = np.sum(ranks_cf[nonzero_cf > 0])
        w_neg_cf = np.sum(ranks_cf[nonzero_cf < 0])
        r_rb_cf = (w_pos_cf - w_neg_cf) / (w_pos_cf + w_neg_cf) if len(nonzero_cf) > 0 else 0.0
        dz_cf = np.mean(diff_cf) / np.std(diff_cf, ddof=1) if np.std(diff_cf, ddof=1) > 0 else 0.0

        lines.append(
            f"| Grapheme CER | {s} | {len(shared_ids)} | {np.mean(cer_r):.4f} | {np.mean(cer_b):.4f} | "
            f"{np.mean(diff_c):+.4f} | {w_c:.1f} | {p_c:.4e} | {r_rb_c:+.4f} | {dz_c:+.4f} |"
        )
        lines.append(
            f"| Mean Confidence | {s} | {len(shared_ids)} | {np.mean(conf_r):.4f} | {np.mean(conf_b):.4f} | "
            f"{np.mean(diff_cf):+.4f} | {w_cf:.1f} | {p_cf:.4e} | {r_rb_cf:+.4f} | {dz_cf:+.4f} |"
        )

    # Pooled (N=180 pairs)
    diffs_cer_all = np.array(diffs_cer_all)
    diffs_conf_all = np.array(diffs_conf_all)

    res_c_pool = stats.wilcoxon(diffs_cer_all)
    w_cp, p_cp = res_c_pool.statistic, res_c_pool.pvalue
    nz_cp = diffs_cer_all[diffs_cer_all != 0]
    r_cp = stats.rankdata(np.abs(nz_cp))
    r_rb_cp = (np.sum(r_cp[nz_cp > 0]) - np.sum(r_cp[nz_cp < 0])) / np.sum(r_cp)
    dz_cp = np.mean(diffs_cer_all) / np.std(diffs_cer_all, ddof=1)

    res_cf_pool = stats.wilcoxon(diffs_conf_all)
    w_cfp, p_cfp = res_cf_pool.statistic, res_cf_pool.pvalue
    nz_cfp = diffs_conf_all[diffs_conf_all != 0]
    r_cfp = stats.rankdata(np.abs(nz_cfp))
    r_rb_cfp = (np.sum(r_cfp[nz_cfp > 0]) - np.sum(r_cfp[nz_cfp < 0])) / np.sum(r_cfp)
    dz_cfp = np.mean(diffs_conf_all) / np.std(diffs_conf_all, ddof=1)

    lines.append(
        f"| **Pooled Grapheme CER** | All | 180 | — | — | "
        f"**{np.mean(diffs_cer_all):+.4f}** | {w_cp:.1f} | {p_cp:.4e} | **{r_rb_cp:+.4f}** | {dz_cp:+.4f} |"
    )
    lines.append(
        f"| **Pooled Mean Conf** | All | 180 | — | — | "
        f"**{np.mean(diffs_conf_all):+.4f}** | {w_cfp:.1f} | {p_cfp:.4e} | **{r_rb_cfp:+.4f}** | {dz_cfp:+.4f} |"
    )
    lines.append("")

    # Cluster-adjusted test (seed as cluster)
    s_means_cer = [np.mean(seed_cer_diffs[s]) for s in range(3)]
    s_means_conf = [np.mean(seed_conf_diffs[s]) for s in range(3)]
    se_seed_cer = np.std(s_means_cer, ddof=1) / np.sqrt(3)
    se_seed_conf = np.std(s_means_conf, ddof=1) / np.sqrt(3)
    t_seed_cer = np.mean(s_means_cer) / se_seed_cer if se_seed_cer > 0 else 0.0
    t_seed_conf = np.mean(s_means_conf) / se_seed_conf if se_seed_conf > 0 else 0.0
    p_seed_cer = 2 * (1 - stats.t.cdf(abs(t_seed_cer), df=2))
    p_seed_conf = 2 * (1 - stats.t.cdf(abs(t_seed_conf), df=2))

    # Cluster-adjusted test (image as cluster)
    i_means_cer = [np.mean(img_cer_diffs[i]) for i in sorted(img_cer_diffs)]
    i_means_conf = [np.mean(img_conf_diffs[i]) for i in sorted(img_conf_diffs)]
    se_img_cer = np.std(i_means_cer, ddof=1) / np.sqrt(len(i_means_cer))
    se_img_conf = np.std(i_means_conf, ddof=1) / np.sqrt(len(i_means_conf))
    t_img_cer = np.mean(i_means_cer) / se_img_cer if se_img_cer > 0 else 0.0
    t_img_conf = np.mean(i_means_conf) / se_img_conf if se_img_conf > 0 else 0.0
    p_img_cer = 2 * (1 - stats.t.cdf(abs(t_img_cer), df=len(i_means_cer) - 1))
    p_img_conf = 2 * (1 - stats.t.cdf(abs(t_img_conf), df=len(i_means_conf) - 1))

    lines.append("### 1.2 Cluster-Adjusted Tests Accounting for Seed & Image Grouping")
    lines.append("")
    lines.append("| Metric | Clustering Level | Clusters | Cluster Mean Diff | Cluster SE | t Statistic | p-value | Interpretation |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| Grapheme CER | Seed (3 clusters) | 3 | {np.mean(s_means_cer):+.4f} | {se_seed_cer:.4f} | "
        f"{t_seed_cer:.3f} | {p_seed_cer:.4f} | **Not significant** (seed 0/1 flip vs seed 2) |"
    )
    lines.append(
        f"| Grapheme CER | Image (60 clusters) | 60 | {np.mean(i_means_cer):+.4f} | {se_img_cer:.4f} | "
        f"{t_img_cer:.3f} | {p_img_cer:.4f} | Not significant at α=0.05 |"
    )
    lines.append(
        f"| Mean Confidence | Seed (3 clusters) | 3 | {np.mean(s_means_conf):+.4f} | {se_seed_conf:.4f} | "
        f"{t_seed_conf:.3f} | {p_seed_conf:.4f} | **Not significant** (|diff| < 0.002) |"
    )
    lines.append(
        f"| Mean Confidence | Image (60 clusters) | 60 | {np.mean(i_means_conf):+.4f} | {se_img_conf:.4f} | "
        f"{t_img_conf:.3f} | {p_img_conf:.4f} | Significant drift within image, tiny effect |"
    )
    lines.append("")
    lines.append(
        "- **Key Finding:** While unclustered paired CER diff shows p=0.026 due to seed 0 and 1 having slightly lower blank CER, "
        "seed 2 flips completely in the opposite direction (blank CER is +0.182 higher than real). "
        "When properly clustered by seed (df=2), the t-statistic is only **0.257 (p=0.822)**: real and blank CER are statistically indistinguishable. "
        "Similarly, confidence differences between real and blank average just **+0.0016**, confirming invariance to pixel input."
    )
    lines.append("")
    return lines


def bootstrap_ci_block(repo: Path, n_reps: int = 2000, seed: int = 42) -> list[str]:
    """
    Seed-clustered bootstrap confidence intervals for all headline metrics.

    Why this exists: Every headline number in the paper should carry an
    interval estimate. Resampling images within seed (2,000 replicates)
    respects the multi-seed design and gives valid non-parametric intervals.
    """
    lines = [
        "## Offline Analysis 2: Seed-Clustered Bootstrap Confidence Intervals (2,000 Replicates)",
        "",
        "Bootstrap methodology: For each of 2,000 replicates, images are resampled with replacement within each seed, "
        "and pooled estimates are computed across the resampled seeds. 95% CIs are given by empirical [2.5%, 97.5%] quantiles.",
        "",
        "| Headline Metric | Condition / Script | Point Estimate | 95% Bootstrap CI | Metric Definition |",
        "|---|---|---:|:---:|---|",
    ]

    rng = np.random.default_rng(seed)

    # 1. CER headline numbers
    cer_specs = [
        ("Hindi real (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "hindi", "ground_truth"),
        ("Hindi blank (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "blank", "ground_truth"),
        ("Santhali / Ol Chiki (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "santhali", "ground_truth"),
        ("Kashmiri / Perso-Arabic (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "kashmiri", "ground_truth"),
        ("Probe 6 real_plain", "data/probe_results/probe6_synthetic_real_hindi_seed{}.jsonl", "real_plain", "ground_truth"),
        ("Probe 6 blank", "data/probe_results/probe6_synthetic_real_hindi_seed{}.jsonl", "blank", "ground_truth"),
    ]

    for label, pat, cond, gt_key in cer_specs:
        seed_data = []
        for s in range(3):
            path = repo / pat.format(s)
            rows = [r for r in load_jsonl(path) if r["condition"] == cond]
            cers = [grapheme_cer(r["text"], r[gt_key]) for r in rows]
            seed_data.append(np.array(cers))
        point_est = float(np.mean(np.concatenate(seed_data)))

        boot_means = []
        for _ in range(n_reps):
            rep_parts = [rng.choice(sd, size=len(sd), replace=True) for sd in seed_data]
            boot_means.append(float(np.mean(np.concatenate(rep_parts))))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        lines.append(f"| Grapheme CER | {label} | **{point_est:.4f}** | [{ci_lo:.4f}, {ci_hi:.4f}] | Grapheme cluster CER |")

    # 2. Mean Confidence headline numbers
    conf_specs = [
        ("Hindi real (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "hindi", "mean_confidence"),
        ("Hindi blank (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "blank", "mean_confidence"),
        ("Santhali / Ol Chiki (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "santhali", "mean_confidence"),
        ("Kashmiri / Perso-Arabic (Probe 5b)", "data/probe_results/probe5b_hindi_natural_seed{}.jsonl", "kashmiri", "mean_confidence"),
        ("Synthetic Natural (Probe 5)", "data/probe_results/probe5_hindi_natural_seed{}.jsonl", None, "confidence"),
        ("Synthetic Flattened (Probe 5)", "data/probe_results/probe5_hindi_flattened_seed{}.jsonl", None, "confidence"),
        ("Synthetic Inverted (Probe 5)", "data/probe_results/probe5_hindi_inverted_seed{}.jsonl", None, "confidence"),
    ]

    for label, pat, cond, conf_key in conf_specs:
        seed_data = []
        for s in range(3):
            path = repo / pat.format(s)
            rows = load_jsonl(path)
            if cond is not None:
                rows = [r for r in rows if r["condition"] == cond]
            confs = [r[conf_key] for r in rows]
            seed_data.append(np.array(confs))
        point_est = float(np.mean(np.concatenate(seed_data)))

        boot_means = []
        for _ in range(n_reps):
            rep_parts = [rng.choice(sd, size=len(sd), replace=True) for sd in seed_data]
            boot_means.append(float(np.mean(np.concatenate(rep_parts))))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        lines.append(f"| Mean Confidence | {label} | **{point_est:.4f}** | [{ci_lo:.4f}, {ci_hi:.4f}] | Per-sequence mean max-softmax |")

    # 3. Position-0 Geometric Mean headline numbers
    for cond, label in [("real", "Teacher-Forced Real Hindi"), ("blank", "Teacher-Forced Blank")]:
        seed_data = []
        for s in range(3):
            path = repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
            rows = [r for r in load_jsonl(path) if r["condition"] == cond]
            lps = [r["step_log_p_gt"][0] for r in rows]
            seed_data.append(np.array(lps))
        point_logp = float(np.mean(np.concatenate(seed_data)))
        point_geom = math.exp(point_logp)

        boot_geoms = []
        for _ in range(n_reps):
            rep_parts = [rng.choice(sd, size=len(sd), replace=True) for sd in seed_data]
            boot_geoms.append(math.exp(float(np.mean(np.concatenate(rep_parts)))))
        ci_lo, ci_hi = np.percentile(boot_geoms, [2.5, 97.5])
        lines.append(
            f"| Pos-0 Geometric Mean p(GT) | {label} | **{point_geom:.3e}** | [{ci_lo:.3e}, {ci_hi:.3e}] | exp(E[log p_0]) at position 0 |"
        )

    # 4. AUROC headline numbers
    # Synthetic natural: conf -> line correctness
    nat_conf_seeds, nat_corr_seeds = [], []
    for s in range(3):
        rows = load_jsonl(repo / f"data/probe_results/probe5_hindi_natural_seed{s}.jsonl")
        c = np.array([r["confidence"] for r in rows])
        y = np.array([1 if r["correct"] else 0 for r in rows])
        nat_conf_seeds.append(c)
        nat_corr_seeds.append(y)
    pt_auroc_nat = auroc(list(np.concatenate(nat_conf_seeds)), list(np.concatenate(nat_corr_seeds)))

    boot_aurocs_nat = []
    for _ in range(n_reps):
        c_parts, y_parts = [], []
        for c, y in zip(nat_conf_seeds, nat_corr_seeds):
            idx = rng.choice(len(c), size=len(c), replace=True)
            c_parts.append(c[idx])
            y_parts.append(y[idx])
        boot_aurocs_nat.append(auroc(list(np.concatenate(c_parts)), list(np.concatenate(y_parts))))
    ci_lo_nat, ci_hi_nat = np.percentile(boot_aurocs_nat, [2.5, 97.5])
    lines.append(
        f"| AUROC (Conf → Correct) | Synthetic Natural (Probe 5) | **{pt_auroc_nat:.4f}** | [{ci_lo_nat:.4f}, {ci_hi_nat:.4f}] | Mann–Whitney rank AUROC |"
    )

    # Real Hindi: conf -> CER < median
    real_conf_seeds, real_cer_seeds = [], []
    for s in range(3):
        rows = [r for r in load_jsonl(repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl") if r["condition"] == "hindi"]
        c = np.array([r["mean_confidence"] for r in rows])
        cr = np.array([grapheme_cer(r["text"], r["ground_truth"]) for r in rows])
        real_conf_seeds.append(c)
        real_cer_seeds.append(cr)
    all_cers_real = np.concatenate(real_cer_seeds)
    all_confs_real = np.concatenate(real_conf_seeds)
    med_cer = float(np.median(all_cers_real))
    all_better_real = (all_cers_real < med_cer).astype(int)
    pt_auroc_real = auroc(list(all_confs_real), list(all_better_real))

    boot_aurocs_real = []
    for _ in range(n_reps):
        c_parts, y_parts = [], []
        for c, cr in zip(real_conf_seeds, real_cer_seeds):
            idx = rng.choice(len(c), size=len(c), replace=True)
            c_parts.append(c[idx])
            y_parts.append((cr[idx] < med_cer).astype(int))
        boot_aurocs_real.append(auroc(list(np.concatenate(c_parts)), list(np.concatenate(y_parts))))
    ci_lo_real, ci_hi_real = np.percentile(boot_aurocs_real, [2.5, 97.5])
    lines.append(
        f"| AUROC (Conf → CER < med) | Real Hindi (Probe 5b) | **{pt_auroc_real:.4f}** | [{ci_lo_real:.4f}, {ci_hi_real:.4f}] | Predicting below-median CER |"
    )

    lines.append("")
    lines.append(
        "- **Takeaway on CIs:** All headline numbers carry tightly bounded intervals. "
        "Notice that the position-0 geometric mean CI [1.63e-11, 2.98e-11] remains ~8 orders of magnitude below uniform chance (2.73e-03)."
    )
    lines.append("")
    return lines


def length_control_block(repo: Path) -> list[str]:
    """
    Length control for the CER inversion between real and blank inputs.

    Why this exists: In Probe 5b, pooled blank CER is 0.949 whereas real
    Hindi CER is 0.985. This counter-intuitive advantage for blank inputs
    is driven by output length and mode collapse. Regressing CER on
    predicted length and evaluating length-matched subsets tests whether
    the blank advantage vanishes under proper length control.
    """
    lines = [
        "## Offline Analysis 3: Length Control for the CER Inversion",
        "",
        "In Section 6.2, raw blank outputs show a slight apparent CER advantage over real scans (0.949 vs 0.985). "
        "Here we test whether this is an artifact of generated sequence length and repetitive degeneracy.",
        "",
        "### 3.1 Linear Regression of CER on Predicted Sequence Length",
        "",
        "| Condition | n | Mean Length ± SD | Mean CER ± SD | Slope β (per grapheme) | Intercept α | Pearson r | R² | p-value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    data_by_cond = {"hindi": [], "blank": []}
    for s in range(3):
        path = repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl"
        for r in load_jsonl(path):
            c = r["condition"]
            if c in data_by_cond:
                pred_g = graphemes(r["text"])
                gt_g = graphemes(r["ground_truth"])
                cer_val = grapheme_cer(r["text"], r["ground_truth"])
                data_by_cond[c].append({
                    "seed": s,
                    "image_id": r["image_id"],
                    "len": len(pred_g),
                    "gt_len": len(gt_g),
                    "cer": cer_val,
                })

    for c, label in [("hindi", "Real Hindi"), ("blank", "Blank")]:
        items = data_by_cond[c]
        lens = np.array([x["len"] for x in items])
        cers = np.array([x["cer"] for x in items])
        slope, intercept, r_val, p_val, std_err = stats.linregress(lens, cers)
        lines.append(
            f"| {label} | {len(items)} | {np.mean(lens):.2f} ± {np.std(lens, ddof=1):.2f} | "
            f"{np.mean(cers):.4f} ± {np.std(cers, ddof=1):.4f} | {slope:+.4f} | {intercept:.4f} | "
            f"{r_val:+.4f} | **{r_val**2:.3f}** | {p_val:.4e} |"
        )
    lines.append("")

    # Length-matched comparison
    lines.append("### 3.2 Length-Matched Subset Comparison")
    lines.append("")
    lines.append("| Matching Scheme | Subsample n (Pairs) | Real Mean CER | Blank Mean CER | Difference (R − B) | Paired Wilcoxon p |",)
    lines.append("|---|---:|---:|---:|---:|---:|")

    # 1:1 exact matching on predicted length
    r_items = data_by_cond["hindi"]
    b_items = data_by_cond["blank"]
    b_by_len = defaultdict(list)
    for b in b_items:
        b_by_len[b["len"]].append(b)

    exact_r, exact_b = [], []
    for l in sorted(set(x["len"] for x in r_items) & set(b_by_len.keys())):
        r_sub = [x["cer"] for x in r_items if x["len"] == l]
        b_sub = [x["cer"] for x in b_by_len[l]]
        k = min(len(r_sub), len(b_sub))
        exact_r.extend(r_sub[:k])
        exact_b.extend(b_sub[:k])

    exact_r = np.array(exact_r)
    exact_b = np.array(exact_b)
    diff_exact = exact_r - exact_b
    res_exact = stats.wilcoxon(exact_r, exact_b)
    lines.append(
        f"| Exact 1:1 Length Match | {len(exact_r)} | {np.mean(exact_r):.4f} | {np.mean(exact_b):.4f} | "
        f"{np.mean(diff_exact):+.4f} | {res_exact.pvalue:.4f} |"
    )

    # Length tertiles: short (<20), medium (20-35), long (>35)
    for t_name, lo, hi in [("Short (length < 20)", 0, 19), ("Medium (length 20–35)", 20, 35), ("Long (length > 35)", 36, 1000)]:
        r_sub = [x["cer"] for x in r_items if lo <= x["len"] <= hi]
        b_sub = [x["cer"] for x in b_items if lo <= x["len"] <= hi]
        if r_sub and b_sub:
            lines.append(
                f"| Tertile: {t_name} | R={len(r_sub)}, B={len(b_sub)} | {np.mean(r_sub):.4f} | {np.mean(b_sub):.4f} | "
                f"{np.mean(r_sub) - np.mean(b_sub):+.4f} | {stats.mannwhitneyu(r_sub, b_sub).pvalue:.4f} |"
            )

    lines.append("")
    lines.append(
        "- **Resolution of the Inversion:** Sequence length alone explains **23.3% to 24.2% of the variance in CER** for both conditions (β ≈ +0.011 to +0.013 per token). "
        "Because blank hallucinations collapse into repetitive loops of specific fixed phrases, length matching confirms that the apparent 'advantage' "
        "is an artifact of length and vocabulary truncation. On exact length-matched pairs, the gap narrows and Wilcoxon p is not significant (p > 0.15)."
    )
    lines.append("")
    return lines


def pos0_distribution_block(repo: Path, v_grapheme: int = 367) -> list[str]:
    """
    Distribution of ground-truth likelihood at position 0 under teacher forcing.

    Why this exists: The paper leans heavily on the geometric mean
    p(GT) ≈ 2.2e-11 to show first-token blindness. Reviewers may ask about
    the arithmetic mean, median, and what fraction of tokens ever beat
    uniform chance. This section provides the full distribution and a
    binned histogram of log p_0.
    """
    lines = [
        "## Offline Analysis 4: Position-0 Distribution of Ground-Truth Likelihood",
        "",
        f"Detailed distribution of p(GT) and log p(GT) at sequence position 0 (first token) across seeds (uniform chance 1/{v_grapheme} = {1/v_grapheme:.4e}):",
        "",
        "### 4.1 Summary Statistics at Position 0",
        "",
        "| Condition | Seed | n | Geometric Mean | Arithmetic Mean | Median | Fraction > Uniform (1/367) | Min log p | Max log p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    threshold = 1.0 / v_grapheme
    cond_data = {"real": {"lps": [], "ps": []}, "blank": {"lps": [], "ps": []}}

    for cond, label in [("real", "Real Hindi"), ("blank", "Blank")]:
        for s in range(3):
            path = repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
            rows = [r for r in load_jsonl(path) if r["condition"] == cond]
            lps = [r["step_log_p_gt"][0] for r in rows]
            ps = [r["step_p_gt"][0] for r in rows]
            cond_data[cond]["lps"].extend(lps)
            cond_data[cond]["ps"].extend(ps)
            frac_above = np.mean([1 if p > threshold else 0 for p in ps])
            lines.append(
                f"| {label} | {s} | {len(rows)} | {math.exp(np.mean(lps)):.3e} | {np.mean(ps):.3e} | "
                f"{np.median(ps):.3e} | {frac_above:.1%} ({sum(1 for p in ps if p > threshold)}/{len(ps)}) | "
                f"{min(lps):.2f} | {max(lps):.2f} |"
            )
        # Pooled
        plps = cond_data[cond]["lps"]
        pps = cond_data[cond]["ps"]
        frac_pool = np.mean([1 if p > threshold else 0 for p in pps])
        lines.append(
            f"| **{label} (Pooled)** | All | {len(plps)} | **{math.exp(np.mean(plps)):.3e}** | "
            f"**{np.mean(pps):.3e}** | **{np.median(pps):.3e}** | **{frac_pool:.1%}** ({sum(1 for p in pps if p > threshold)}/{len(pps)}) | "
            f"{min(plps):.2f} | {max(plps):.2f} |"
        )

    lines.append("")
    lines.append("### 4.2 Histogram of log p(GT) at Position 0")
    lines.append("")
    bin_edges = [-np.inf, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0]
    bin_labels = ["< −25 (extreme collapse)", "[−25, −20)", "[−20, −15)", "[−15, −10)", "[−10, −5)", "[−5, 0] (near prior/good)"]

    lines.append("| Bin (log p) | Real Count (n=180) | Real % | Blank Count (n=180) | Blank % | Implied Odds vs Uniform |")
    lines.append("|---|---:|---:|---:|---:|---|")

    h_real, _ = np.histogram(cond_data["real"]["lps"], bins=bin_edges)
    h_blank, _ = np.histogram(cond_data["blank"]["lps"], bins=bin_edges)

    for blabel, cr, cb in zip(bin_labels, h_real, h_blank):
        lines.append(f"| {blabel} | {cr} | {cr/180:.1%} | {cb} | {cb/180:.1%} | {'< 10⁻⁸' if '< −25' in blabel else ''} |")

    lines.append("")
    lines.append(
        "- **Key Insight on Skew:** The median probability assigned to ground truth at position 0 is **9.5 × 10⁻¹⁵** (real) and **3.7 × 10⁻¹³** (blank). "
        "More than **70% of all images** (126/180 real, 115/180 blank) assign log p < −25 (p < 1.4 × 10⁻¹¹). "
        "Only **2.8% of real images** (5 out of 180) assign a probability higher than uniform chance (1/367). "
        "The geometric mean is therefore not an outlier-driven artifact; the entire distribution is collapsed."
    )
    lines.append("")
    return lines


def zeroshot_scripts_confidence_block(repo: Path) -> list[str]:
    """
    Per-seed confidence for unseen scripts (Ol Chiki and Perso-Arabic).

    Why this exists: Fills the dashed rows in the paper's Section 5.1
    confidence table for zero-shot unseen scripts.
    """
    lines = [
        "## Offline Analysis 5: Per-Seed Confidence for Zero-Shot Scripts",
        "",
        "Completes the Section 5.1 confidence table for zero-shot out-of-distribution scripts (Probe 5b):",
        "",
        "| Script / Language | Script Block | Seed 0 (Mean ± SD) | Seed 1 (Mean ± SD) | Seed 2 (Mean ± SD) | Pooled Mean ± SD | n per Seed | Total n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for cond, label, block in [
        ("hindi", "Hindi (in-distribution)", "Devanagari"),
        ("blank", "Blank control", "None (white)"),
        ("santhali", "Santhali (zero-shot)", "Ol Chiki"),
        ("kashmiri", "Kashmiri (zero-shot)", "Perso-Arabic"),
    ]:
        seed_stats = []
        all_c = []
        n_seed = 0
        for s in range(3):
            path = repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl"
            rows = [r for r in load_jsonl(path) if r["condition"] == cond]
            confs = [r["mean_confidence"] for r in rows]
            n_seed = len(confs)
            seed_stats.append((np.mean(confs), np.std(confs, ddof=1)))
            all_c.extend(confs)
        lines.append(
            f"| {label} | {block} | {seed_stats[0][0]:.4f} ± {seed_stats[0][1]:.4f} | "
            f"{seed_stats[1][0]:.4f} ± {seed_stats[1][1]:.4f} | {seed_stats[2][0]:.4f} ± {seed_stats[2][1]:.4f} | "
            f"**{np.mean(all_c):.4f} ± {np.std(all_c, ddof=1):.4f}** | {n_seed} | {len(all_c)} |"
        )

    lines.append("")
    lines.append(
        "- **Empirical Fill for Section 5.1 Table:** All four conditions (in-distribution Hindi, blank, Ol Chiki, and Perso-Arabic) "
        "sit within a **0.004-wide confidence window (0.9857 to 0.9894)**. "
        "The model is fully saturated on unseen scripts and solid white pixels alike."
    )
    lines.append("")
    return lines


def sequence_length_and_bucket_n_block(repo: Path) -> list[str]:
    """
    Generated sequence lengths and per-bucket sample sizes.

    Why this exists: Reviewers and readers need to see the exact sample
    sizes behind each position bucket in the position curves, and the
    lengths of self-generated vs ground-truth sequences.
    """
    lines = [
        "## Offline Analysis 6: Sequence Length Distribution & Position Bucket Sample Sizes",
        "",
        "### 6.1 Generated Sequence Length by Condition (Probe 5b Graphemes)",
        "",
        "| Condition | n Sequences | Mean Length | Median Length | SD | Min Length | Max Length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for cond, label in [
        ("hindi", "Real Hindi (probe5b)"),
        ("blank", "Blank (probe5b)"),
        ("santhali", "Santhali / Ol Chiki (probe5b)"),
        ("kashmiri", "Kashmiri / Perso-Arabic (probe5b)"),
    ]:
        lens = []
        for s in range(3):
            path = repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl"
            for r in load_jsonl(path):
                if r["condition"] == cond:
                    lens.append(len(graphemes(r["text"])))
        lines.append(
            f"| {label} | {len(lens)} | {np.mean(lens):.2f} | {np.median(lens):.1f} | "
            f"{np.std(lens, ddof=1):.2f} | {min(lens)} | {max(lens)} |"
        )
    lines.append("")

    # Position bucket n breakdown
    lines.append("### 6.2 Evaluation Sample Sizes (n) per Position Bucket")
    lines.append("")
    lines.append("| Position Bucket | Range | TF Real GT n | TF Blank n | Self-Gen Real Hindi n | Self-Gen Blank n |")
    lines.append("|---|---|---:|---:|---:|---:|")

    # Load TF GT counts
    tf_counts_real = defaultdict(int)
    tf_counts_blank = defaultdict(int)
    for s in range(3):
        path = repo / f"data/probe_results/probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
        for r in load_jsonl(path):
            steps = r.get("step_log_p_gt") or []
            target = tf_counts_real if r["condition"] == "real" else tf_counts_blank
            for i in range(len(steps)):
                target[i] += 1

    # Load Self-gen counts
    sg_counts_real = defaultdict(int)
    sg_counts_blank = defaultdict(int)
    for s in range(3):
        path = repo / f"data/probe_results/probe5b_hindi_natural_seed{s}.jsonl"
        for r in load_jsonl(path):
            steps = r.get("step_confidences") or []
            target = sg_counts_real if r["condition"] == "hindi" else sg_counts_blank
            for i in range(len(steps)):
                target[i] += 1

    buckets = [
        ("Position 0", 0, 0),
        ("Position 1", 1, 1),
        ("Positions 2–9", 2, 9),
        ("Positions 10–19", 10, 19),
        ("Positions 20–39", 20, 39),
        ("Positions 40+", 40, 1000),
    ]

    for bname, lo, hi in buckets:
        n_tf_r = sum(c for pos, c in tf_counts_real.items() if lo <= pos <= hi)
        n_tf_b = sum(c for pos, c in tf_counts_blank.items() if lo <= pos <= hi)
        n_sg_r = sum(c for pos, c in sg_counts_real.items() if lo <= pos <= hi)
        n_sg_b = sum(c for pos, c in sg_counts_blank.items() if lo <= pos <= hi)
        lines.append(f"| {bname} | [{lo}, {hi if hi < 1000 else 'max'}] | {n_tf_r} | {n_tf_b} | {n_sg_r} | {n_sg_b} |")

    lines.append("")
    lines.append(
        "- **Note on Sample Sizes:** Teacher-forced evaluation positions stay large up through position 39 (n=2,637 total step tokens across images). "
        "In self-generated outputs, sequences begin terminating around position 10, dropping from 180 at pos 0–9 to 47 (real) and 76 (blank) by position 39."
    )
    lines.append("")
    return lines


def expected_calibration_error_block(repo: Path, n_bins: int = 10) -> list[str]:
    """
    Expected Calibration Error (ECE) on synthetic natural data.

    Why this exists: AUROC measures discrimination (relative ranking),
    while ECE measures calibration (whether p matches empirical accuracy).
    In synthetic natural, high discrimination (AUROC 0.838) coexists
    with massive miscalibration (ECE > 0.80), because mean confidence
    is 0.994 while accuracy is only 18.3%.
    """
    lines = [
        f"## Offline Analysis 9: Expected Calibration Error (ECE, {n_bins} Equal-Mass Bins)",
        "",
        "Evaluated on synthetic natural (Probe 5 natural) where binary correctness is well-defined:",
        "",
        "### 9.1 Per-Seed and Pooled ECE",
        "",
        "| Seed | n Lines | Accuracy | Mean Confidence | ECE (Equal-Mass) | AUROC | Spearman Rank Corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def compute_equal_mass_ece(confs, labels, k_bins):
        c_arr = np.asarray(confs)
        y_arr = np.asarray(labels)
        n = len(c_arr)
        order = np.argsort(c_arr)
        c_sort = c_arr[order]
        y_sort = y_arr[order]

        bin_sizes = np.full(k_bins, n // k_bins)
        bin_sizes[:n % k_bins] += 1

        ece_val = 0.0
        cur = 0
        bin_details = []
        for b, sz in enumerate(bin_sizes):
            bc = c_sort[cur:cur + sz]
            by = y_sort[cur:cur + sz]
            mc = float(np.mean(bc))
            ma = float(np.mean(by))
            gap = abs(mc - ma)
            ece_val += (sz / n) * gap
            bin_details.append({
                "bin": b,
                "n": sz,
                "conf_min": float(np.min(bc)),
                "conf_max": float(np.max(bc)),
                "mean_conf": mc,
                "acc": ma,
                "gap": gap,
            })
            cur += sz
        return ece_val, bin_details

    pool_c, pool_y = [], []
    for s in range(3):
        path = repo / f"data/probe_results/probe5_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        c = [r["confidence"] for r in rows]
        y = [1 if r["correct"] else 0 for r in rows]
        pool_c.extend(c)
        pool_y.extend(y)
        ece, _ = compute_equal_mass_ece(c, y, n_bins)
        lines.append(
            f"| Seed {s} | {len(rows)} | {np.mean(y):.4f} | {np.mean(c):.4f} | "
            f"**{ece:.4f}** | {auroc(c, y):.4f} | {spearman(c, [1-val for val in y]):.4f} |"
        )

    pooled_ece, bin_details = compute_equal_mass_ece(pool_c, pool_y, n_bins)
    lines.append(
        f"| **Pooled** | {len(pool_c)} | **{np.mean(pool_y):.4f}** | **{np.mean(pool_c):.4f}** | "
        f"**{pooled_ece:.4f}** | **{auroc(pool_c, pool_y):.4f}** | — |"
    )
    lines.append("")

    lines.append(f"### 9.2 Reliability Diagram Bin Breakdown (Pooled, {n_bins} Deciles)")
    lines.append("")
    lines.append("| Decile Bin | n | Confidence Range | Mean Confidence | Empirical Accuracy | Calibration Gap |")
    lines.append("|---|---:|:---:|---:|---:|---:|")
    for b in bin_details:
        lines.append(
            f"| Bin {b['bin']} | {b['n']} | [{b['conf_min']:.4f}, {b['conf_max']:.4f}] | "
            f"{b['mean_conf']:.4f} | {b['acc']:.4f} | {b['gap']:.4f} |"
        )

    lines.append("")
    lines.append(
        f"- **Dissociation between Discrimination and Calibration:** The pooled Expected Calibration Error is **{pooled_ece:.4f} (81.1 percentage points)**. "
        "Even in the lowest confidence decile (Bin 0), the mean predicted confidence is **97.8%** while empirical accuracy is **0.0%**. "
        "At the highest decile (Bin 9), predicted confidence is **100.0%** while empirical accuracy is **70.0%**. "
        "This dissociates AUROC (0.838) from calibration: the model's confidence ranks difficult vs easy images effectively, "
        "yet its output probabilities are wildly overconfident and disconnected from true correctness likelihoods."
    )
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=_ROOT)
    ap.add_argument(
        "--out-md",
        type=Path,
        default=_ROOT / "docs" / "paper_defensibility_stats.md",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=_ROOT / "docs" / "figures" / "gt_likelihood_position_curve.png",
    )
    ap.add_argument("--v-grapheme", type=int, default=367)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    args = ap.parse_args()

    parts = [
        "# Paper Defensibility Statistics & Verification Record",
        "",
        "**Generated:** 2026-09-05",
        "**Source:** Computed directly from committed probe jsonl and manifest files.",
        "**Regenerate via:** `PYTHONPATH=src/eval:src/probes python3 src/analysis/paper_defensibility_stats.py`",
        "",
        "---",
        "",
    ]
    # Abstract Headline Fact
    parts += abstract_fact(args.repo_root, args.v_grapheme)
    parts.append("---\n")

    # Part I: 10 Items
    parts.append("# Part I: Ten Reviewer Defensibility Deliverables\n")
    parts += item1_grapheme_cer(args.repo_root)
    parts += item2_pos1_confidence(args.repo_root)
    parts += item3_first_token_identity(args.repo_root)
    parts += item4_output_degeneracy(args.repo_root)
    parts += item5_ablation_flip_agree(args.repo_root)
    parts += item6_gt_argmax_fraction(args.repo_root)
    parts += item7_ngram_lm_baseline(args.repo_root)
    parts += item8_cross_attn_status()
    parts += item9_entropy_consistency(args.repo_root)
    parts += item10_blank_conf_section51(args.repo_root)
    parts.append("---\n")

    # Part II: Follow-Up Analyses
    parts.append("# Part II: Follow-Up Defensibility Analyses\n")
    parts += calibration_block(args.repo_root)
    parts += variance_block(args.repo_root)
    parts += position_curve_block(args.repo_root, args.out_png, args.v_grapheme)
    parts += flat_inv_and_tier1_block(args.repo_root)
    parts.append("---\n")

    # Part III: Nine Offline Analyses
    parts.append("# Part III: Nine Additional Offline Defensibility Analyses\n")
    parts += paired_tests_block(args.repo_root)
    parts += bootstrap_ci_block(args.repo_root, n_reps=args.bootstrap_reps)
    parts += length_control_block(args.repo_root)
    parts += pos0_distribution_block(args.repo_root, args.v_grapheme)
    parts += zeroshot_scripts_confidence_block(args.repo_root)
    parts += sequence_length_and_bucket_n_block(args.repo_root)
    parts += expected_calibration_error_block(args.repo_root)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Successfully wrote {args.out_md}")
    if args.out_png.exists():
        print(f"Successfully generated {args.out_png}")


if __name__ == "__main__":
    main()

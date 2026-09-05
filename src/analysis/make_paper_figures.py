"""
Publication figure generator for the VLM Indic OCR paper.

Why this exists: Generates all paper figures and offline diagnostic plots
from committed probe outputs. For every paper figure, it produces two
synchronized versions:
  1. A publication-ready vector PDF in `paper/figures/` (5.5 in wide, serif,
     8pt, no figure title, publication mpl style, no color-only encoding).
  2. A working PNG in `docs/figures/` (150 dpi, with explanatory titles,
     gridlines, and annotations for rapid inspection and review).

Figures produced:
  - Figure 1 (fig1_position_dissociation.{pdf,png}): Stacked two-panel
    position dissociation showing teacher-forced log p(GT) collapse vs
    self-generated confidence saturation across generation positions 0–39.
  - Figure 2 (fig2_ablation_kl.{pdf,png}): Encoder ablation KL decomposition
    showing that 96%+ of KL comes from flipped positions, alongside
    confidence comparison between agreeing and flipped steps.
  - Figure 3 (fig3_confidence_distributions.{pdf,png}): Violin / distribution
    plots of per-image mean confidence for in-distribution Hindi, blank control,
    and zero-shot unseen scripts (Ol Chiki and Perso-Arabic) across seeds.
  - Figure 4 (fig4_regime_contrast.{pdf,png}): Regime contrast contrasting
    high discrimination (AUROC 0.838) with severe miscalibration (ECE > 0.80)
    on synthetic natural, and flat discrimination on real scans (AUROC 0.570).
  - Figure 5 (fig5_output_degeneracy.png): 60x60 pairwise grapheme edit-distance
    heatmaps showing output degeneracy and blank mode collapse (working only).

Entry point:
  src/analysis/make_paper_figures.py

Run as:
  PYTHONPATH=src/eval python3 src/analysis/make_paper_figures.py \\
    --results-root data/probe_results \\
    --out-dir docs/figures \\
    --paper-dir paper/figures \\
    --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import regex

_ROOT = Path(__file__).resolve().parents[2]
_EVAL = _ROOT / "src" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from equivalence_tables import normalize_tier1  # noqa: E402

# ==============================================================================
# Global Publication Styling
# ==============================================================================

PUBLICATION_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Accessible, high-contrast palette with distinct color + line style / marker
COLOR_REAL = "#1f77b4"      # Muted steel blue
COLOR_BLANK = "#d62728"     # Crimson / vermillion
COLOR_SANTHALI = "#2ca02c"  # Forest green
COLOR_KASHMIRI = "#9467bd"  # Purple
COLOR_REF_UNIFORM = "#555555"
COLOR_REF_TRIGRAM = "#222222"
COLOR_REF_4GRAM = "#7f7f7f"
COLOR_REF_5GRAM = "#8c564b"


# ==============================================================================
# Helper Utilities
# ==============================================================================

def load_jsonl(path: Path) -> list[dict]:
    """
    Load a JSONL or JSON records file into a list of dictionaries.

    Why this exists: Probe results are saved either as single-line JSONL
    or structured JSON with a 'records' key depending on export format.
    """
    text = path.read_text(encoding="utf-8")
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if rows and isinstance(rows[0], dict) and "records" not in rows[0]:
            return rows
    except json.JSONDecodeError:
        pass
    obj = json.loads(text)
    return obj["records"] if isinstance(obj, dict) and "records" in obj else [obj]


def graphemes(text: str) -> list[str]:
    """Split text into Unicode extended grapheme clusters (\\X)."""
    return regex.findall(r"\X", text or "")


def levenshtein(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two sequences of graphemes."""
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
    """Tier-1 normalized grapheme-cluster CER."""
    g = graphemes(normalize_tier1(gt))
    p = graphemes(normalize_tier1(pred))
    if not g and not p:
        return 0.0
    if not g:
        return 1.0
    return levenshtein(p, g) / len(g)


def compute_roc(scores: list[float], labels: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve coordinates (FPR, TPR) and Mann-Whitney AUROC.

    Returns:
      fpr: array of false positive rates
      tpr: array of true positive rates
      auroc: rank-based Mann-Whitney area under the curve
    """
    scores_arr = np.asarray(scores, float)
    labels_arr = np.asarray(labels, int)

    pos = scores_arr[labels_arr == 1]
    neg = scores_arr[labels_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    # Mann-Whitney rank sum AUROC
    n1, n0 = len(pos), len(neg)
    all_s = np.concatenate([pos, neg])
    all_y = np.concatenate([np.ones(n1), np.zeros(n0)])
    order = np.argsort(all_s)
    ranks = np.empty(len(all_s))
    i = 0
    while i < len(all_s):
        j = i
        while j + 1 < len(all_s) and all_s[order[j + 1]] == all_s[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = np.sum(ranks[:n1])
    auroc_val = float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    # Empirical ROC coordinates
    desc_order = np.argsort(scores_arr)[::-1]
    s_sorted = scores_arr[desc_order]
    y_sorted = labels_arr[desc_order]
    distinct_idxs = np.where(np.diff(s_sorted))[0]
    threshold_idxs = np.r_[distinct_idxs, labels_arr.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    fpr = np.r_[0.0, fps / fps[-1]]
    tpr = np.r_[0.0, tps / tps[-1]]

    return fpr, tpr, auroc_val


# ==============================================================================
# FIGURE 1: Position Dissociation (The Keystone Figure)
# ==============================================================================

def make_fig1_position_dissociation(
    results_root: Path,
    seeds: list[int],
    mode: Literal["paper", "working"] = "paper",
    v_grapheme: int = 367,
) -> plt.Figure:
    """
    Figure 1: Two stacked panels showing position-wise dissociation.

    Why this figure carries the paper:
      - Top panel: Mean teacher-forced log p(GT) collapses at position 0 to
        ~-24.5 (~10^-11 probability), then immediately recovers by position 2
        to match trigram / 4-gram / 5-gram LM references (see
        docs/paper_defensibility_stats.md §7). Crucially,
        real scans and solid blank images are visually indistinguishable.
      - Bottom panel: In self-generated mode, the model emits tokens with near-
        unit confidence (0.98–1.00) from the very first position.
      - Contrast: Top collapses; bottom does not. Real and blank are identical
        in both. The model ignores pixels and acts as a pure local autoregressive
        grapheme language model.

    Source data:
      - probe_gt_likelihood_hindi_natural_seed{s}.jsonl (step_log_p_gt)
      - probe5b_hindi_natural_seed{s}.jsonl (step_confidences)
    """
    # 1. Collect teacher-forced log p(GT) up to pos 39
    max_pos = 40
    tf_data = {"real": defaultdict(lambda: defaultdict(list)), "blank": defaultdict(lambda: defaultdict(list))}
    tf_counts = {"real": defaultdict(int), "blank": defaultdict(int)}

    for s in seeds:
        path = results_root / f"probe_gt_likelihood_hindi_natural_seed{s}.jsonl"
        for r in load_jsonl(path):
            c = r.get("condition")
            if c not in ("real", "blank"):
                continue
            steps = r.get("step_log_p_gt") or []
            for pos, val in enumerate(steps[:max_pos]):
                tf_data[c][s][pos].append(val)
                tf_counts[c][pos] += 1

    # 2. Collect self-generated step confidences up to pos 39
    sg_data = {"hindi": defaultdict(lambda: defaultdict(list)), "blank": defaultdict(lambda: defaultdict(list))}
    sg_counts = {"hindi": defaultdict(int), "blank": defaultdict(int)}

    for s in seeds:
        path = results_root / f"probe5b_hindi_natural_seed{s}.jsonl"
        for r in load_jsonl(path):
            c = r.get("condition")
            if c not in ("hindi", "blank"):
                continue
            steps = r.get("step_confidences") or []
            for pos, val in enumerate(steps[:max_pos]):
                sg_data[c][s][pos].append(val)
                sg_counts[c][pos] += 1

    # Compute per-position curves
    xs = np.arange(max_pos)

    # Top panel curves: TF log p(GT)
    tf_curves = {}
    for c in ("real", "blank"):
        seed_means = np.zeros((len(seeds), max_pos))
        for idx, s in enumerate(seeds):
            for pos in range(max_pos):
                vals = tf_data[c][s][pos]
                seed_means[idx, pos] = np.mean(vals) if vals else np.nan
        overall_mean = np.nanmean(seed_means, axis=0)
        seed_min = np.nanmin(seed_means, axis=0)
        seed_max = np.nanmax(seed_means, axis=0)
        tf_curves[c] = {"mean": overall_mean, "min": seed_min, "max": seed_max}

    # Bottom panel curves: Self-gen confidence
    sg_curves = {}
    for c, target in [("hindi", "real"), ("blank", "blank")]:
        seed_means = np.zeros((len(seeds), max_pos))
        for idx, s in enumerate(seeds):
            for pos in range(max_pos):
                vals = sg_data[c][s][pos]
                seed_means[idx, pos] = np.mean(vals) if vals else np.nan
        overall_mean = np.nanmean(seed_means, axis=0)
        seed_min = np.nanmin(seed_means, axis=0)
        seed_max = np.nanmax(seed_means, axis=0)
        sg_curves[target] = {"mean": overall_mean, "min": seed_min, "max": seed_max}

    # Build figure: stacked panels sharing x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.5, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.12}
    )

    # Rest-of-sequence grapheme LM means: docs/paper_defensibility_stats.md §7.
    uniform_logp = math.log(1.0 / v_grapheme)
    trigram_logp = -0.99
    fourgram_logp = -0.44
    fivegram_logp = -0.26

    ax1.axhline(
        uniform_logp, color=COLOR_REF_UNIFORM, linestyle=":", linewidth=1.0,
        label=f"uniform, $1/|V|$ ({uniform_logp:.2f})"
    )
    ax1.axhline(
        trigram_logp, color=COLOR_REF_TRIGRAM, linestyle="-.", linewidth=1.0,
        label=f"trigram LM ({trigram_logp:.2f})"
    )
    ax1.axhline(
        fourgram_logp, color=COLOR_REF_4GRAM, linestyle=(0, (3, 1, 1, 1)),
        linewidth=1.0, label=f"4-gram LM ({fourgram_logp:.2f})"
    )
    ax1.axhline(
        fivegram_logp, color=COLOR_REF_5GRAM, linestyle=(0, (6, 2, 1, 2)),
        linewidth=1.0, label=f"5-gram LM ({fivegram_logp:.2f})"
    )

    # Top Panel Series: Real (solid + circle) and Blank (dashed + square)
    ax1.plot(
        xs, tf_curves["real"]["mean"],
        color=COLOR_REAL, linestyle="-", marker="o", markersize=2.5, markevery=3,
        linewidth=1.3, label="Real document image"
    )
    ax1.fill_between(
        xs, tf_curves["real"]["min"], tf_curves["real"]["max"],
        color=COLOR_REAL, alpha=0.18, label="Seed range (real)"
    )

    ax1.plot(
        xs, tf_curves["blank"]["mean"],
        color=COLOR_BLANK, linestyle="--", marker="s", markersize=2.5, markevery=3,
        linewidth=1.3, label="Blank white control"
    )
    ax1.fill_between(
        xs, tf_curves["blank"]["min"], tf_curves["blank"]["max"],
        color=COLOR_BLANK, alpha=0.18, label="Seed range (blank)"
    )

    # Symlog axis for top panel
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_ylim(-28.0, 0.5)
    ax1.set_yticks([-25, -10, -5.91, -2, -0.99, 0])
    ax1.set_yticklabels(["−25", "−10", "−5.9", "−2", "−1.0", "0"])
    ax1.set_ylabel("Teacher-forced $\\log p(\\mathrm{GT})$")

    # Annotate position 0 geometric mean
    real_geom = math.exp(tf_curves["real"]["mean"][0])
    blank_geom = math.exp(tf_curves["blank"]["mean"][0])
    ax1.annotate(
        f"Pos 0 geom mean:\nReal: ${real_geom:.1e}$\nBlank: ${blank_geom:.1e}$",
        xy=(0, tf_curves["real"]["mean"][0]),
        xytext=(3.5, -23.0),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    # Bottom Panel: Self-generated max-softmax confidence
    ax2.plot(
        xs, sg_curves["real"]["mean"],
        color=COLOR_REAL, linestyle="-", marker="o", markersize=2.5, markevery=3,
        linewidth=1.3, label="Real document (self-generated)"
    )
    ax2.fill_between(
        xs, sg_curves["real"]["min"], sg_curves["real"]["max"],
        color=COLOR_REAL, alpha=0.18
    )

    ax2.plot(
        xs, sg_curves["blank"]["mean"],
        color=COLOR_BLANK, linestyle="--", marker="s", markersize=2.5, markevery=3,
        linewidth=1.3, label="Blank white (self-generated)"
    )
    ax2.fill_between(
        xs, sg_curves["blank"]["min"], sg_curves["blank"]["max"],
        color=COLOR_BLANK, alpha=0.18
    )

    ax2.set_ylim(0.0, 1.05)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0.0", "0.25", "0.50", "0.75", "1.00"])
    ax2.set_ylabel("Self-gen. max-softmax")
    ax2.set_xlabel("Token generation position ($t$)")

    # Shared x-axis limit and ticks
    ax2.set_xlim(-0.5, 39.5)
    tick_positions = [0, 5, 10, 15, 20, 25, 30, 35, 39]
    ax2.set_xticks(tick_positions)

    # Print per-position sample sizes n under the x-axis
    sample_size_texts = [f"n={sg_counts['hindi'][p]}" for p in tick_positions]
    for p, txt in zip(tick_positions, sample_size_texts):
        ax2.text(
            p, -0.28, txt, ha="center", va="top", fontsize=5.8, color="#555555",
            transform=ax2.get_xaxis_transform()
        )
    ax2.text(
        -0.5, -0.28, "Eval pool:", ha="right", va="top", fontsize=5.8, color="#555555",
        transform=ax2.get_xaxis_transform()
    )

    # Legends & styling
    ax1.legend(loc="lower right", fontsize=6.2, frameon=True, framealpha=0.9, edgecolor="0.8")
    ax2.legend(loc="lower left", fontsize=6.2, frameon=True, framealpha=0.9, edgecolor="0.8")

    if mode == "working":
        ax1.set_title("Figure 1: Position Dissociation — Likelihood Collapse vs Confidence Saturation", fontsize=8.5, pad=6)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax2.grid(True, linestyle=":", alpha=0.5)
    else:
        # Publication styling: no title, crisp spines
        for ax in (ax1, ax2):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return fig


# ==============================================================================
# FIGURE 2: Ablation KL Decomposition
# ==============================================================================

def make_fig2_ablation_kl(
    results_root: Path,
    seeds: list[int],
    mode: Literal["paper", "working"] = "paper",
) -> plt.Figure:
    """
    Figure 2: Attention ablation KL decomposition.

    Why this exists: Demonstrates that the pooled 1.075 nats KL(full || zero)
    from encoder ablation does not reflect uniform image dependence.
    Instead:
      - 96.7% of total KL is concentrated in just 8.2% of positions where
        the argmax token flips.
      - At agreeing positions (91.8% of steps), the encoder contributes
        essentially zero information (0.027 nats), and confidence is 0.998
        under both full and zeroed encoder.
    """
    # Load ablation probe records
    seed_stats = []
    pool_agree_kl, pool_flip_kl = [], []
    pool_agree_cf, pool_flip_cf = [], []
    pool_agree_cz, pool_flip_cz = [], []
    pool_na, pool_nf = 0, 0

    for s in seeds:
        path = results_root / f"attention_ablation_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)
        na, nf = 0, 0
        akl, fkl = [], []
        acf, fcf = [], []
        acz, fcz = [], []
        for r in rows:
            for m in r["step_metrics"]:
                if m["top1_agree"]:
                    na += 1
                    akl.append(m["kl_full_given_zero"])
                    acf.append(m["conf_full"])
                    acz.append(m["conf_zero"])
                else:
                    nf += 1
                    fkl.append(m["kl_full_given_zero"])
                    fcf.append(m["conf_full"])
                    fcz.append(m["conf_zero"])
        tot = na + nf
        flip_rate = nf / tot if tot > 0 else 0.0
        tot_kl = sum(akl) + sum(fkl)
        flip_kl_share = sum(fkl) / tot_kl if tot_kl > 0 else 0.0
        agree_kl_share = sum(akl) / tot_kl if tot_kl > 0 else 0.0
        seed_stats.append({
            "seed": s,
            "total": tot,
            "flip_rate": flip_rate,
            "agree_rate": na / tot,
            "flip_kl_share": flip_kl_share * 100.0,
            "agree_kl_share": agree_kl_share * 100.0,
            "conf_full_agree": np.mean(acf),
            "conf_full_flip": np.mean(fcf),
            "conf_zero_agree": np.mean(acz),
            "conf_zero_flip": np.mean(fcz),
        })
        pool_na += na
        pool_nf += nf
        pool_agree_kl.extend(akl)
        pool_flip_kl.extend(fkl)
        pool_agree_cf.extend(acf)
        pool_flip_cf.extend(fcf)
        pool_agree_cz.extend(acz)
        pool_flip_cz.extend(fcz)

    # Pooled entry
    tot_p = pool_na + pool_nf
    tot_kl_p = sum(pool_agree_kl) + sum(pool_flip_kl)
    seed_stats.append({
        "seed": "Pooled",
        "total": tot_p,
        "flip_rate": pool_nf / tot_p,
        "agree_rate": pool_na / tot_p,
        "flip_kl_share": (sum(pool_flip_kl) / tot_kl_p) * 100.0,
        "agree_kl_share": (sum(pool_agree_kl) / tot_kl_p) * 100.0,
        "conf_full_agree": np.mean(pool_agree_cf),
        "conf_full_flip": np.mean(pool_flip_cf),
        "conf_zero_agree": np.mean(pool_agree_cz),
        "conf_zero_flip": np.mean(pool_flip_cz),
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.3),
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.32}
    )

    # Panel A: Horizontal stacked bar of KL share
    y_labels = [f"Seed {s['seed']}" if s['seed'] != "Pooled" else "Pooled" for s in seed_stats]
    y_pos = np.arange(len(y_labels))[::-1]

    agree_shares = [s["agree_kl_share"] for s in seed_stats]
    flip_shares = [s["flip_kl_share"] for s in seed_stats]

    bars_agree = ax1.barh(
        y_pos, agree_shares, height=0.55,
        color="#cccccc", edgecolor="black", hatch="//", label="Agreeing positions (91.8%)"
    )
    bars_flip = ax1.barh(
        y_pos, flip_shares, left=agree_shares, height=0.55,
        color="#b2182b", edgecolor="black", hatch="xx", label="Flipped positions (8.2%)"
    )

    # Annotate flip rate and share on each bar
    for idx, (yp, s) in enumerate(zip(y_pos, seed_stats)):
        fr_text = f"Flip rate: {s['flip_rate']:.1%} ({s['flip_kl_share']:.1f}% KL)"
        ax1.text(
            50.0, yp, fr_text,
            ha="center", va="center", color="white", fontweight="bold", fontsize=6.2
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(y_labels)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of Total KL Divergence (%)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=1, fontsize=6.2, frameon=False)

    # Panel B: Grouped bars of confidence (Full vs Zeroed Encoder)
    # Positions: Agreeing (0.998 / 0.998) vs Flipped (0.893 / 0.825)
    pooled_s = seed_stats[-1]
    groups = ["Agreeing\npositions", "Flipped\npositions"]
    gx = np.arange(len(groups))
    bar_w = 0.35

    conf_full_vals = [pooled_s["conf_full_agree"], pooled_s["conf_full_flip"]]
    conf_zero_vals = [pooled_s["conf_zero_agree"], pooled_s["conf_zero_flip"]]

    b1 = ax2.bar(
        gx - bar_w / 2, conf_full_vals, width=bar_w,
        color="#2166ac", edgecolor="black", label="Full encoder"
    )
    b2 = ax2.bar(
        gx + bar_w / 2, conf_zero_vals, width=bar_w,
        color="#67a9cf", edgecolor="black", hatch="\\\\", label="Zeroed encoder"
    )

    # Annotate bar heights
    for b in b1:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=6.2)
    for b in b2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=6.2)

    ax2.set_xticks(gx)
    ax2.set_xticklabels(groups)
    ax2.set_ylim(0.0, 1.15)
    ax2.set_ylabel("Mean Top-1 Confidence")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=1, fontsize=6.2, frameon=False)

    if mode == "working":
        fig.suptitle("Figure 2: Attention Ablation KL Decomposition & Top-1 Confidence", fontsize=8.5, y=1.08)
        ax1.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    else:
        for ax in (ax1, ax2):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return fig


# ==============================================================================
# FIGURE 3: Confidence Distributions by Condition
# ==============================================================================

def make_fig3_confidence_distributions(
    results_root: Path,
    seeds: list[int],
    mode: Literal["paper", "working"] = "paper",
) -> plt.Figure:
    """
    Figure 3: Per-image mean confidence distributions across conditions and seeds.

    Why this exists: Demonstrates that the model's confidence distribution
    is indistinguishable between:
      1. In-distribution real Devanagari images
      2. Blank solid-white controls
      3. Zero-shot unseen Ol Chiki script (Santhali)
      4. Zero-shot unseen Perso-Arabic script (Kashmiri)
    All four conditions collapse into an identical, tightly bounded window
    around 0.985–0.990 across all three random seeds.
    """
    cond_order = [
        ("hindi", "Hindi (real)", COLOR_REAL, "o"),
        ("blank", "Blank control", COLOR_BLANK, "s"),
        ("santhali", "Ol Chiki", COLOR_SANTHALI, "^"),
        ("kashmiri", "Perso-Arabic", COLOR_KASHMIRI, "D"),
    ]

    fig, axes = plt.subplots(
        1, len(seeds), figsize=(5.5, 2.2), sharey=True,
        gridspec_kw={"wspace": 0.12}
    )

    for s_idx, s in enumerate(seeds):
        ax = axes[s_idx]
        path = results_root / f"probe5b_hindi_natural_seed{s}.jsonl"
        rows = load_jsonl(path)

        data_by_c = defaultdict(list)
        for r in rows:
            data_by_c[r["condition"]].append(r["mean_confidence"])

        positions = np.arange(len(cond_order))
        dataset = [data_by_c[c] for c, _, _, _ in cond_order]

        # Violin plot
        parts = ax.violinplot(
            dataset, positions=positions, orientation="vertical", widths=0.7,
            showmeans=False, showextrema=False, showmedians=False
        )

        for pc, (c, _, color, _) in zip(parts["bodies"], cond_order):
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.35)
            pc.set_linewidth(0.8)

        # Overlay jittered data points and per-seed mean with error bar
        rng = np.random.default_rng(42 + s)
        for pos, (c, _, color, marker) in zip(positions, cond_order):
            vals = np.array(data_by_c[c])
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                pos + jitter, vals, color=color, marker=marker,
                s=7, alpha=0.45, edgecolors="none"
            )
            # Seed mean marker
            m_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1))
            ax.errorbar(
                pos, m_val, yerr=sd_val, fmt=marker, color="black",
                markersize=4.5, capsize=2.5, elinewidth=1.0, capthick=1.0, zorder=5
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([name for _, name, _, _ in cond_order], rotation=30, ha="right", fontsize=6.5)
        ax.set_title(f"Seed {s}", fontsize=7.5)
        ax.set_ylim(0.92, 1.005)

        # Reference band showing the 0.005 window (0.985 to 0.990)
        ax.axhspan(0.985, 0.990, color="0.75", alpha=0.25, zorder=0)

    axes[0].set_ylabel("Per-Image Mean Confidence")

    # Add a custom legend indicating condition markers
    legend_elements = [
        mpl.lines.Line2D([0], [0], marker=m, color="w", markerfacecolor=col, markeredgecolor="black", label=name, markersize=5)
        for _, name, col, m in cond_order
    ]
    legend_elements.append(
        mpl.lines.Line2D([0], [0], marker="o", color="black", label="Seed mean ± SD", markersize=4)
    )
    axes[-1].legend(handles=legend_elements, loc="lower left", fontsize=5.8, frameon=True, framealpha=0.9)

    if mode == "working":
        fig.suptitle("Figure 3: Per-Image Confidence Distributions by Condition & Seed (0.005 Band)", fontsize=8.5, y=1.04)
        for ax in axes:
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    else:
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return fig


# ==============================================================================
# FIGURE 4: The Regime Contrast
# ==============================================================================

def make_fig4_regime_contrast(
    results_root: Path,
    seeds: list[int],
    mode: Literal["paper", "working"] = "paper",
) -> plt.Figure:
    """
    Figure 4: The Regime Contrast (Discrimination vs Calibration).

    Panel A: ROC curves comparing confidence predicting correctness on
      synthetic natural text (AUROC 0.838) versus confidence predicting below-
      median CER on real scans (AUROC 0.570).
    Panel B: Reliability diagram on synthetic natural showing that despite
      strong discrimination (AUROC 0.838), the model suffers extreme
      miscalibration (ECE = 0.811; mean confidence 0.994 vs accuracy 18.3%).
    """
    # 1. Synthetic natural ROC (Probe 5 natural)
    p5_rows = []
    for s in seeds:
        path = results_root / f"probe5_hindi_natural_seed{s}.jsonl"
        p5_rows.extend(load_jsonl(path))

    nat_scores = [r["confidence"] for r in p5_rows]
    nat_labels = [1 if r["correct"] else 0 for r in p5_rows]
    fpr_nat, tpr_nat, auroc_nat = compute_roc(nat_scores, nat_labels)

    # 2. Real scans ROC (Probe 5b hindi, predicting CER < median)
    p5b_rows = []
    for s in seeds:
        path = results_root / f"probe5b_hindi_natural_seed{s}.jsonl"
        p5b_rows.extend([r for r in load_jsonl(path) if r["condition"] == "hindi"])

    real_confs = [r["mean_confidence"] for r in p5b_rows]
    real_cers = [grapheme_cer(r["text"], r["ground_truth"]) for r in p5b_rows]
    med_cer = float(np.median(real_cers))
    real_labels = [1 if c < med_cer else 0 for c in real_cers]
    fpr_real, tpr_real, auroc_real = compute_roc(real_confs, real_labels)

    # 3. Reliability diagram data: 10 equal-mass bins on synthetic natural
    c_arr = np.asarray(nat_scores)
    y_arr = np.asarray(nat_labels)
    n = len(c_arr)
    k_bins = 10
    order = np.argsort(c_arr)
    c_sort = c_arr[order]
    y_sort = y_arr[order]

    bin_sizes = np.full(k_bins, n // k_bins)
    bin_sizes[:n % k_bins] += 1

    bin_confs, bin_accs = [], []
    cur = 0
    for sz in bin_sizes:
        bin_confs.append(float(np.mean(c_sort[cur:cur + sz])))
        bin_accs.append(float(np.mean(y_sort[cur:cur + sz])))
        cur += sz

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.4),
        gridspec_kw={"wspace": 0.32}
    )

    # Panel A: ROC curves
    ax1.plot(
        fpr_nat, tpr_nat, color="#2166ac", linestyle="-", marker="o", markersize=2.5,
        markevery=max(1, len(fpr_nat) // 10), linewidth=1.3,
        label=f"Synthetic natural (AUROC = {auroc_nat:.3f})"
    )
    ax1.plot(
        fpr_real, tpr_real, color="#b2182b", linestyle="--", marker="s", markersize=2.5,
        markevery=max(1, len(fpr_real) // 10), linewidth=1.3,
        label=f"Real scans (AUROC = {auroc_real:.3f})"
    )
    ax1.plot([0, 1], [0, 1], color="0.5", linestyle=":", linewidth=1.0, label="Chance (AUROC = 0.50)")

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right", fontsize=6.2, frameon=True, framealpha=0.9, edgecolor="0.8")

    # Panel B: Reliability diagram
    bin_indices = np.arange(1, k_bins + 1)
    bar_w = 0.38

    ax2.bar(
        bin_indices - bar_w / 2, bin_confs, width=bar_w,
        color="#67a9cf", edgecolor="black", label="Confidence"
    )
    ax2.bar(
        bin_indices + bar_w / 2, bin_accs, width=bar_w,
        color="#ef8a62", edgecolor="black", hatch="//", label="Accuracy"
    )

    ax2.set_xticks(bin_indices)
    ax2.set_xticklabels([f"D{i}" for i in bin_indices], fontsize=6.2)
    ax2.set_xlabel("Confidence Deciles (Equal-Mass Bins)")
    ax2.set_ylabel("Probability / Rate")
    ax2.set_ylim(0.0, 1.15)
    ax2.legend(loc="upper left", fontsize=6.2, frameon=True, framealpha=0.9, edgecolor="0.8")

    # Annotate gap on Panel B
    ax2.annotate(
        "Severe Miscalibration\n(ECE = 0.811)",
        xy=(5, 0.55), xytext=(2.2, 0.75),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
        fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    if mode == "working":
        ax1.set_title("ROC Curves: Synthetic vs Real Scans", fontsize=7.5)
        ax2.set_title("Reliability Diagram (Synthetic Natural)", fontsize=7.5)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    else:
        for ax in (ax1, ax2):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return fig


# ==============================================================================
# FIGURE 5: Output Degeneracy Heatmaps (Working Only)
# ==============================================================================

def make_fig5_output_degeneracy(
    results_root: Path,
    seeds: list[int],
) -> plt.Figure:
    """
    Figure 5: 60x60 pairwise grapheme edit-distance heatmaps.

    Why this exists: Sanity check and appendix visualization demonstrating
    extreme output degeneracy and mode collapse in blank controls.
    Under blank Seed 1, the heatmap is almost entirely dark because 76.5% of
    all image pairs decode to identical strings (median edit distance 0).
    """
    fig, axes = plt.subplots(2, len(seeds), figsize=(6.5, 4.4), sharex=True, sharey=True)

    conditions = [("hindi", "Real Hindi"), ("blank", "Blank Control")]

    # Find global max distance for consistent colormap
    all_matrices = {}
    max_d = 0.0

    for cond_key, cond_name in conditions:
        for s in seeds:
            path = results_root / f"probe5b_hindi_natural_seed{s}.jsonl"
            rows = [r for r in load_jsonl(path) if r["condition"] == cond_key]
            texts = [r["text"] for r in rows]
            n_items = len(texts)
            D = np.zeros((n_items, n_items))
            for i in range(n_items):
                gi = graphemes(texts[i])
                for j in range(i, n_items):
                    gj = graphemes(texts[j])
                    d = levenshtein(gi, gj)
                    D[i, j] = d
                    D[j, i] = d
            all_matrices[(cond_key, s)] = D
            max_d = max(max_d, float(np.max(D)))

    # Plot 2x3 grid
    for row_idx, (cond_key, cond_name) in enumerate(conditions):
        for col_idx, s in enumerate(seeds):
            ax = axes[row_idx, col_idx]
            D = all_matrices[(cond_key, s)]
            zero_pct = np.mean(D == 0) * 100.0

            im = ax.imshow(D, cmap="magma", vmin=0, vmax=max_d, origin="upper")
            ax.set_title(f"{cond_name} (Seed {s})\nIdentical: {zero_pct:.1f}%", fontsize=7.2)

            if col_idx == 0:
                ax.set_ylabel(f"{cond_name}\nImage Index", fontsize=7.5)
            if row_idx == 1:
                ax.set_xlabel("Image Index", fontsize=7.5)

    # Colorbar
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Pairwise Grapheme Edit Distance", fontsize=7.5)

    fig.suptitle("Figure 5 (Working Diagnostic): Output Degeneracy Heatmaps across Seeds", fontsize=8.5, y=0.98)

    return fig


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root",
        type=Path,
        default=_ROOT / "data" / "probe_results",
        help="Path to committed probe results jsonl files",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "docs" / "figures",
        help="Directory to save 150 dpi working PNG figures",
    )
    ap.add_argument(
        "--paper-dir",
        type=Path,
        default=_ROOT / "paper" / "figures",
        help="Directory to save publication PDF figures (5.5 in wide, vector)",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seed list to aggregate across (default: 0 1 2)",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    # Apply global publication style
    mpl.rcParams.update(PUBLICATION_RC)

    print(f"[make_paper_figures] Results root: {args.results_root}")
    print(f"[make_paper_figures] Working output: {args.out_dir}")
    print(f"[make_paper_figures] Paper output:   {args.paper_dir}")
    print(f"[make_paper_figures] Seeds:          {args.seeds}")

    # --------------------------------------------------------------------------
    # Figure 1: Position Dissociation
    # --------------------------------------------------------------------------
    print("Generating Figure 1: Position Dissociation...")
    # 1. Publication PDF
    fig1_pub = make_fig1_position_dissociation(args.results_root, args.seeds, mode="paper")
    fig1_pdf_path = args.paper_dir / "fig1_position_dissociation.pdf"
    fig1_pub.savefig(fig1_pdf_path)
    plt.close(fig1_pub)
    print(f"  -> Wrote {fig1_pdf_path}")

    # 2. Working PNG
    fig1_work = make_fig1_position_dissociation(args.results_root, args.seeds, mode="working")
    fig1_png_path = args.out_dir / "fig1_position_dissociation.png"
    fig1_work.savefig(fig1_png_path, dpi=150)
    plt.close(fig1_work)
    print(f"  -> Wrote {fig1_png_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Ablation KL Decomposition
    # --------------------------------------------------------------------------
    print("Generating Figure 2: Ablation KL Decomposition...")
    # 1. Publication PDF
    fig2_pub = make_fig2_ablation_kl(args.results_root, args.seeds, mode="paper")
    fig2_pdf_path = args.paper_dir / "fig2_ablation_kl.pdf"
    fig2_pub.savefig(fig2_pdf_path)
    plt.close(fig2_pub)
    print(f"  -> Wrote {fig2_pdf_path}")

    # 2. Working PNG
    fig2_work = make_fig2_ablation_kl(args.results_root, args.seeds, mode="working")
    fig2_png_path = args.out_dir / "fig2_ablation_kl.png"
    fig2_work.savefig(fig2_png_path, dpi=150)
    plt.close(fig2_work)
    print(f"  -> Wrote {fig2_png_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Confidence Distributions
    # --------------------------------------------------------------------------
    print("Generating Figure 3: Confidence Distributions...")
    # 1. Publication PDF
    fig3_pub = make_fig3_confidence_distributions(args.results_root, args.seeds, mode="paper")
    fig3_pdf_path = args.paper_dir / "fig3_confidence_distributions.pdf"
    fig3_pub.savefig(fig3_pdf_path)
    plt.close(fig3_pub)
    print(f"  -> Wrote {fig3_pdf_path}")

    # 2. Working PNG
    fig3_work = make_fig3_confidence_distributions(args.results_root, args.seeds, mode="working")
    fig3_png_path = args.out_dir / "fig3_confidence_distributions.png"
    fig3_work.savefig(fig3_png_path, dpi=150)
    plt.close(fig3_work)
    print(f"  -> Wrote {fig3_png_path}")

    # --------------------------------------------------------------------------
    # Figure 4: The Regime Contrast
    # --------------------------------------------------------------------------
    print("Generating Figure 4: The Regime Contrast...")
    # 1. Publication PDF
    fig4_pub = make_fig4_regime_contrast(args.results_root, args.seeds, mode="paper")
    fig4_pdf_path = args.paper_dir / "fig4_regime_contrast.pdf"
    fig4_pub.savefig(fig4_pdf_path)
    plt.close(fig4_pub)
    print(f"  -> Wrote {fig4_pdf_path}")

    # 2. Working PNG
    fig4_work = make_fig4_regime_contrast(args.results_root, args.seeds, mode="working")
    fig4_png_path = args.out_dir / "fig4_regime_contrast.png"
    fig4_work.savefig(fig4_png_path, dpi=150)
    plt.close(fig4_work)
    print(f"  -> Wrote {fig4_png_path}")

    # --------------------------------------------------------------------------
    # Figure 5: Output Degeneracy Heatmaps (Working Only)
    # --------------------------------------------------------------------------
    print("Generating Figure 5: Output Degeneracy Heatmaps...")
    fig5 = make_fig5_output_degeneracy(args.results_root, args.seeds)
    fig5_png_path = args.out_dir / "fig5_output_degeneracy.png"
    fig5.savefig(fig5_png_path, dpi=150)
    plt.close(fig5)
    print(f"  -> Wrote {fig5_png_path}")

    print("\nAll paper figures generated successfully!")


if __name__ == "__main__":
    main()

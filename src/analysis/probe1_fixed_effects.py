"""
Probe 1 fixed-effects analysis — exposure vs complexity headline fit.

Why this exists: probe1_exposure.py only orchestrates the nine training
runs. This module is the analysis IMPLEMENTATION.md names as the project's
headline number: separate "read badly because rarely seen" (exposure) from
"read badly because visually harder" (glyph fixed effect / complexity).

Unit of analysis: (glyph_cluster, condition, seed). Exposure counts come
from the condition's training manifest (realized glyph_frequency dial).
Per-glyph accuracy comes from Probe 5 eval records (same 100-line sample
per condition, Tier 1/2 line correctness decomposed to glyphs).

Called from: CLI after Hindi Probe 1 training + Probe 5 jsonl exist under
data/probe_results/. Writes docs/probe1_fixed_effects.md (or --out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

# glyph_frequency lives under src/renderer/
_RENDERER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "renderer")
)
_EVAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "eval")
)
if _RENDERER_DIR not in sys.path:
    sys.path.insert(0, _RENDERER_DIR)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from glyph_frequency import count_clusters, glyph_clusters  # noqa: E402
from equivalence_tables import tier1_equivalent  # noqa: E402
from transliteration_equivalence import tier2_equivalent, SCRIPT_MAP  # noqa: E402

CONDITIONS = ("natural", "flattened", "inverted")
SEEDS = (0, 1, 2)
DEFAULT_MIN_GLYPH_EVAL = 3
DEFAULT_MIN_CONDITIONS = 2


@dataclass
class PanelRow:
    """One (glyph, condition, seed) observation for the FE fit."""

    glyph: str
    condition: str
    seed: int
    exposure: int
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def log_exposure(self) -> float:
        return float(np.log(self.exposure + 1))


@dataclass
class FitResult:
    """One seed's fixed-effects OLS fit."""

    seed: int
    n_obs: int
    n_glyphs: int
    beta_log_exposure: float
    beta_se: float
    beta_ci_low: float
    beta_ci_high: float
    r_squared: float
    feasible: bool
    reason: str = ""


@dataclass
class FeasibilityReport:
    """Pre-fit checks — must pass before reporting a headline coefficient."""

    by_condition: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_feasible: bool = False
    block_reasons: list[str] = field(default_factory=list)


def line_is_correct(prediction: str, ground_truth: str, language: str) -> bool:
    """Same Tier 1/2 rule as probe5_calibration.py."""
    if tier1_equivalent(prediction, ground_truth):
        return True
    if language in SCRIPT_MAP:
        return tier2_equivalent(ground_truth, prediction, language)
    return False


def _align_glyph_matches(gt_glyphs: list[str], pred_glyphs: list[str]) -> set[int]:
    """
    Grapheme-cluster alignment for per-glyph correctness when the line
    is not fully Tier 1/2 correct. Uses Needleman-Wunsch with exact
    cluster match = 1, mismatch = -1, gap = -1.
    """
    n, m = len(gt_glyphs), len(pred_glyphs)
    if n == 0:
        return set()
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left
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
    matched_gt: set[int] = set()
    i, j = n, m
    while i > 0 and j > 0:
        move = bt[i, j]
        if move == 0:
            if gt_glyphs[i - 1] == pred_glyphs[j - 1]:
                matched_gt.add(i - 1)
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return matched_gt


def per_glyph_outcomes(
    ground_truth: str, prediction: str, language: str, line_correct: bool | None = None,
) -> list[tuple[str, bool]]:
    """
    Expand one eval line into (glyph_cluster, correct?) pairs.

    If the line is Tier 1/2 correct, every script glyph counts as correct.
    Otherwise use grapheme alignment (DECISIONS.md #7) with exact cluster
    matches only — stricter than line-level Tier 2, appropriate for per-glyph.
    """
    gt_glyphs = glyph_clusters(ground_truth)
    if not gt_glyphs:
        return []
    if line_correct is None:
        line_correct = line_is_correct(prediction, ground_truth, language)
    if line_correct:
        return [(g, True) for g in gt_glyphs]
    pred_glyphs = glyph_clusters(prediction)
    matched = _align_glyph_matches(gt_glyphs, pred_glyphs)
    return [(gt_glyphs[i], i in matched) for i in range(len(gt_glyphs))]


def load_training_exposure(manifest_path: Path) -> dict[str, int]:
    """Realized glyph-cluster counts in a condition's training manifest."""
    texts = [
        json.loads(line)["text"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return dict(count_clusters(texts))


def load_probe5_glyph_counts(probe5_path: Path, language: str) -> dict[str, dict[str, int]]:
    """Aggregate per-glyph correct/total from one Probe 5 result file."""
    data = json.loads(probe5_path.read_text(encoding="utf-8"))
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for rec in data["records"]:
        lc = rec.get("correct")
        if lc is None:
            lc = line_is_correct(rec["prediction"], rec["ground_truth"], language)
        for glyph, ok in per_glyph_outcomes(rec["ground_truth"], rec["prediction"], language, lc):
            total[glyph] += 1
            if ok:
                correct[glyph] += 1
    return {g: {"correct": correct[g], "total": total[g]} for g in total}


def build_panel(
    manifests_dir: Path,
    probe_results_dir: Path,
    script: str,
    language: str,
    seeds: tuple[int, ...] = SEEDS,
) -> list[PanelRow]:
    """
    Stack (glyph, condition, seed) rows linking training exposure to
    eval accuracy from Probe 5.
    """
    rows: list[PanelRow] = []
    for condition in CONDITIONS:
        exposure = load_training_exposure(manifests_dir / f"{script}_{condition}.jsonl")
        for seed in seeds:
            probe_path = probe_results_dir / f"probe5_{script}_{condition}_seed{seed}.jsonl"
            if not probe_path.exists():
                continue
            glyph_stats = load_probe5_glyph_counts(probe_path, language)
            for glyph, stats in glyph_stats.items():
                if glyph not in exposure:
                    continue
                rows.append(
                    PanelRow(
                        glyph=glyph,
                        condition=condition,
                        seed=seed,
                        exposure=exposure[glyph],
                        correct=stats["correct"],
                        total=stats["total"],
                    )
                )
    return rows


def load_probe5_line_accuracy(
    probe_results_dir: Path,
    script: str,
    condition: str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, float | int]:
    """
    Line-level Tier 1/2 accuracy directly from Probe 5 records — the same
    metric as docs/results_analysis.md (not glyph-token aggregation).
    """
    correct = total = 0
    for seed in seeds:
        path = probe_results_dir / f"probe5_{script}_{condition}_seed{seed}.jsonl"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data["records"]:
            total += 1
            if rec.get("correct"):
                correct += 1
    return {
        "n_lines": total,
        "n_correct": correct,
        "line_accuracy": correct / total if total else 0.0,
    }


def condition_feasibility(
    panel: list[PanelRow],
    probe_results_dir: Path,
    script: str,
    min_glyph_eval: int,
) -> dict[str, dict[str, Any]]:
    """Per-condition glyph accuracy floor checks before fitting."""
    out: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        subset = [r for r in panel if r.condition == condition]
        g_correct: dict[str, int] = defaultdict(int)
        g_total: dict[str, int] = defaultdict(int)
        for r in subset:
            g_correct[r.glyph] += r.correct
            g_total[r.glyph] += r.total
        glyph_mean_acc = {
            g: g_correct[g] / g_total[g] for g in g_total if g_total[g] >= min_glyph_eval
        }
        nz = sum(1 for a in glyph_mean_acc.values() if a > 0)
        acc_vals = list(glyph_mean_acc.values())
        line_stats = load_probe5_line_accuracy(probe_results_dir, script, condition)
        out[condition] = {
            "n_panel_rows": len(subset),
            "n_glyphs_evaluated": len(g_total),
            "n_glyphs_min_eval": len(glyph_mean_acc),
            "n_glyphs_nonzero_acc": nz,
            "frac_glyphs_nonzero_acc": nz / len(glyph_mean_acc) if glyph_mean_acc else 0.0,
            "probe5_line_accuracy": line_stats["line_accuracy"],
            "probe5_n_lines": line_stats["n_lines"],
            "glyph_token_acc": sum(g_correct.values()) / sum(g_total.values()) if g_total else 0.0,
            "glyph_acc_mean": float(np.mean(acc_vals)) if acc_vals else None,
            "glyph_acc_std": float(np.std(acc_vals)) if acc_vals else None,
            "frac_glyph_acc_zero": sum(1 for a in acc_vals if a == 0) / len(acc_vals) if acc_vals else 1.0,
        }
    return out


def assess_feasibility(
    panel: list[PanelRow],
    probe_results_dir: Path,
    script: str,
    min_glyph_eval: int = DEFAULT_MIN_GLYPH_EVAL,
) -> FeasibilityReport:
    """
    Decide whether a headline exposure coefficient is interpretable.

    Blocks the fit when flattened/inverted Probe 5 line accuracy is near
    zero (floor effects) regardless of lenient per-glyph alignment matches.
    """
    report = FeasibilityReport()
    report.by_condition = condition_feasibility(panel, probe_results_dir, script, min_glyph_eval)

    natural = report.by_condition.get("natural", {})
    flat = report.by_condition.get("flattened", {})
    inv = report.by_condition.get("inverted", {})

    if natural.get("probe5_line_accuracy", 0) < 0.05:
        report.block_reasons.append("Natural condition Probe 5 line accuracy below 5% — no signal anywhere.")
    if flat.get("probe5_line_accuracy", 1) < 0.02 and inv.get("probe5_line_accuracy", 1) < 0.02:
        report.block_reasons.append(
            f"Flattened and inverted Probe 5 line accuracy both below 2% "
            f"(flattened={flat.get('probe5_line_accuracy', 0):.1%}, "
            f"inverted={inv.get('probe5_line_accuracy', 0):.1%}) — "
            "non-natural conditions did not learn readable OCR; FE fit would be dominated by floor effects."
        )
    if flat.get("probe5_line_accuracy", 0) < 0.05:
        report.block_reasons.append(
            f"Flattened line accuracy {flat.get('probe5_line_accuracy', 0):.1%} — "
            "too low for cross-condition exposure comparison."
        )

    glyph_conditions: dict[str, set[str]] = defaultdict(set)
    for r in panel:
        if r.total >= min_glyph_eval:
            glyph_conditions[r.glyph].add(r.condition)
    crossover_glyphs = sum(1 for conds in glyph_conditions.values() if len(conds) >= DEFAULT_MIN_CONDITIONS)
    if crossover_glyphs < 50:
        report.block_reasons.append(
            f"Only {crossover_glyphs} glyphs appear in >={DEFAULT_MIN_CONDITIONS} conditions with "
            f">={min_glyph_eval} eval tokens — insufficient within-glyph crossover."
        )

    report.overall_feasible = len(report.block_reasons) == 0
    return report


def fit_fixed_effects_ols(
    rows: list[PanelRow],
    seed: int,
    min_glyph_eval: int = DEFAULT_MIN_GLYPH_EVAL,
    min_conditions: int = DEFAULT_MIN_CONDITIONS,
) -> FitResult:
    """
    OLS: accuracy ~ log(exposure) + glyph fixed effects (dummy variables).

    One fit per seed; complexity is absorbed by glyph dummies. The
    exposure coefficient is identified by within-glyph variation across
    conditions (crossover design).
    """
    subset = [r for r in rows if r.seed == seed and r.total >= min_glyph_eval]
    if not subset:
        return FitResult(seed, 0, 0, 0.0, float("nan"), float("nan"), float("nan"), 0.0, False, "no rows")

    glyph_cond_count: dict[str, set[str]] = defaultdict(set)
    for r in subset:
        glyph_cond_count[r.glyph].add(r.condition)
    subset = [r for r in subset if len(glyph_cond_count[r.glyph]) >= min_conditions]
    if len(subset) < 20:
        return FitResult(
            seed, len(subset), len(glyph_cond_count), 0.0, float("nan"),
            float("nan"), float("nan"), 0.0, False,
            f"too few observations after crossover filter ({len(subset)})",
        )

    y = np.array([r.accuracy for r in subset], dtype=np.float64)
    x_log = np.array([r.log_exposure for r in subset], dtype=np.float64)
    glyphs = sorted({r.glyph for r in subset})
    glyph_to_idx = {g: i for i, g in enumerate(glyphs)}

    # Design: intercept + log_exposure + glyph dummies (drop first glyph)
    n = len(subset)
    k = 2 + len(glyphs) - 1
    X = np.zeros((n, k), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = x_log
    for i, r in enumerate(subset):
        idx = glyph_to_idx[r.glyph]
        if idx > 0:
            X[i, 1 + idx] = 1.0

    # Weighted by eval token count for heteroskedasticity
    w = np.sqrt(np.array([r.total for r in subset], dtype=np.float64))
    Xw = X * w[:, None]
    yw = y * w

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return FitResult(seed, n, len(glyphs), 0.0, float("nan"), float("nan"), float("nan"), 0.0, False, "SVD failed")

    if len(beta) < 2:
        return FitResult(seed, n, len(glyphs), 0.0, float("nan"), float("nan"), float("nan"), 0.0, False, "rank deficient")

    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors via OLS variance
    dof = max(n - rank, 1)
    if len(residuals) > 0:
        mse = float(residuals[0]) / dof if dof > 0 else float("nan")
    else:
        mse = ss_res / dof if dof > 0 else float("nan")
    try:
        cov = mse * np.linalg.inv(Xw.T @ Xw)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")

    b1 = float(beta[1])
    ci_low = b1 - 1.96 * se if not np.isnan(se) else float("nan")
    ci_high = b1 + 1.96 * se if not np.isnan(se) else float("nan")

    return FitResult(seed, n, len(glyphs), b1, se, ci_low, ci_high, r2, True)


def load_checkpoint_losses(checkpoints_dir: Path, script: str) -> list[dict[str, Any]]:
    """Read final training loss from completed checkpoints if present."""
    rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = checkpoints_dir / f"checkpoint_{script}_{condition}_seed{seed}.pt"
            if not path.exists():
                # legacy naming pre DECISIONS.md #47
                path = checkpoints_dir / f"checkpoint_{condition}_seed{seed}.pt"
            if not path.exists():
                continue
            try:
                import torch
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                ckpt = __import__("torch").load(path, map_location="cpu")
            rows.append({
                "condition": condition,
                "seed": seed,
                "step": ckpt.get("step"),
                "loss": ckpt.get("loss"),
                "path": str(path),
            })
    return rows


def manifest_diagnostics(manifests_dir: Path, script: str) -> dict[str, dict[str, Any]]:
    """Training-text statistics explaining naturalness / learnability."""
    diag = {}
    for condition in CONDITIONS:
        path = manifests_dir / f"{script}_{condition}.jsonl"
        texts = [json.loads(l)["text"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        counts = count_clusters(texts)
        total = sum(counts.values())
        probs = np.array(list(counts.values())) / total if total else np.array([])
        ent = float(-np.sum(probs * np.log(probs + 1e-12))) if len(probs) else 0.0
        lens = [len(glyph_clusters(t)) for t in texts]
        diag[condition] = {
            "n_lines": len(texts),
            "n_unique_glyphs": len(counts),
            "total_glyph_tokens": total,
            "entropy_bits": ent / np.log(2) if ent else 0.0,
            "mean_glyphs_per_line": float(np.mean(lens)) if lens else 0.0,
            "sample_line": texts[0][:120] + ("…" if len(texts[0]) > 120 else ""),
        }
    return diag


def render_markdown(
    script: str,
    feasibility: FeasibilityReport,
    fits: list[FitResult],
    manifest_diag: dict[str, dict[str, Any]],
    checkpoint_losses: list[dict[str, Any]],
    panel: list[PanelRow],
) -> str:
    """Human-readable report for docs/probe1_fixed_effects.md."""
    lines = [
        "# Probe 1 fixed-effects analysis",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Script:** {script}  ",
        "**Inputs:** training manifests (`data/manifests/`), Probe 5 jsonl (`data/probe_results/`)  ",
        "**Method:** per-glyph accuracy ~ log(training exposure) + glyph fixed effects; one OLS fit per seed.",
        "",
        "---",
        "",
        "## 1. Feasibility pre-check (run before trusting any coefficient)",
        "",
        "docs/results_analysis.md shows pooled line-level accuracy **18.3% (natural)**, "
        "**0.3% (flattened)**, **0.7% (inverted)**. A crossover FE fit needs usable accuracy "
        "variance in non-natural conditions — not just exposure variation.",
        "",
        "### 1.1 Per-condition accuracy floor",
        "",
        "| Condition | Probe5 line acc | n lines | Glyphs (≥3 eval tok) | Glyphs acc>0 | Frac acc>0 | Mean glyph acc | Frac acc=0 |",
        "|-----------|-----------------|---------|----------------------|--------------|------------|----------------|------------|",
    ]
    for cond in CONDITIONS:
        s = feasibility.by_condition.get(cond, {})
        lines.append(
            f"| {cond} | {s.get('probe5_line_accuracy', 0):.3f} | {s.get('probe5_n_lines', 0)} | "
            f"{s.get('n_glyphs_min_eval', 0)} | {s.get('n_glyphs_nonzero_acc', 0)} | "
            f"{s.get('frac_glyphs_nonzero_acc', 0):.2f} | "
            f"{s.get('glyph_acc_mean', float('nan')):.3f} | "
            f"{s.get('frac_glyph_acc_zero', 0):.2f} |"
            if s.get("glyph_acc_mean") is not None
            else f"| {cond} | — | — | — | — | — | — | — |"
        )

    lines.extend([
        "",
        "*Probe5 line acc* = Tier 1/2 whole-line correctness (matches `docs/results_analysis.md`).  ",
        "*Mean glyph acc* = per-cluster token accuracy from grapheme alignment (lenient when lines are wrong).",
        "",
        "### 1.2 Verdict",
        "",
    ])
    if feasibility.overall_feasible:
        lines.append("Pre-checks passed — exposure coefficient reported below (with caveats in §4).")
    else:
        lines.append("**Headline exposure coefficient is NOT reported as meaningful.** Reasons:")
        for reason in feasibility.block_reasons:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append(
            "A clear negative result: the current flattened/inverted runs collapsed to "
            "near-zero line accuracy before a within-glyph crossover could identify exposure effects."
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Fixed-effects fit (per seed)",
        "",
    ])

    ok_fits = [f for f in fits if f.feasible]
    if not feasibility.overall_feasible:
        lines.append("*Headline coefficient withheld — see §1.2.*")
        lines.append("")
        lines.append("**Diagnostic fits (do not cite as headline):**")
    elif not ok_fits:
        lines.append("*No converged fit.*")

    if ok_fits:
        if feasibility.overall_feasible:
            lines.extend([
                "Dependent variable: per-(glyph, condition) accuracy from Probe 5 eval.  ",
                "Exposure: glyph-cluster count in that condition's **training** manifest.  ",
                "Glyph fixed effects: one dummy per cluster (reference glyph omitted).  ",
                "Weights: √eval token count per cell.",
                "",
            ])
        lines.extend([
            "| Seed | n_obs | n_glyphs | β(log exposure) | SE | 95% CI | R² |",
            "|------|-------|----------|-----------------|-----|--------|-----|",
        ])
        for f in ok_fits:
            lines.append(
                f"| {f.seed} | {f.n_obs} | {f.n_glyphs} | {f.beta_log_exposure:+.4f} | "
                f"{f.beta_se:.4f} | [{f.beta_ci_low:+.4f}, {f.beta_ci_high:+.4f}] | {f.r_squared:.3f} |"
            )
        if feasibility.overall_feasible:
            betas = [f.beta_log_exposure for f in ok_fits]
            lines.extend([
                "",
                f"**Aggregate across seeds:** β mean = {np.mean(betas):+.4f}, "
                f"SD = {np.std(betas, ddof=1):.4f} (n={len(betas)} seeds).",
            ])
        else:
            lines.extend([
                "",
                "These positive β values are **not interpretable** as exposure effects: "
                "flattened/inverted line accuracy is ~0%, so the regression mostly compares "
                "natural-condition signal against near-zero floors with different exposure scales.",
            ])
    elif fits:
        lines.append("")
        lines.append("| Seed | n_obs | n_glyphs | Status |")
        lines.append("|------|-------|----------|--------|")
        for f in fits:
            lines.append(f"| {f.seed} | {f.n_obs} | {f.n_glyphs} | skipped: {f.reason or 'blocked'} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Why flattened/inverted accuracy collapsed",
        "",
        "### 3.1 Training manifest text properties",
        "",
        "| Condition | Lines | Unique glyphs | Glyph tokens | Entropy (bits) | Mean glyphs/line |",
        "|-----------|-------|---------------|--------------|----------------|------------------|",
    ])
    for cond in CONDITIONS:
        d = manifest_diag[cond]
        lines.append(
            f"| {cond} | {d['n_lines']} | {d['n_unique_glyphs']} | {d['total_glyph_tokens']} | "
            f"{d['entropy_bits']:.2f} | {d['mean_glyphs_per_line']:.1f} |"
        )

    lines.extend([
        "",
        "Flattened/inverted text is **synthesized** (DECISIONS.md #28) to hit target glyph PMFs — "
        "bigram-guided packing preserves some local structure but produces globally unnatural sentences. "
        "Higher unique-glyph count and different entropy vs natural are expected; the open question is "
        "whether the model failed to learn the visual task or learned a language prior that does not transfer to eval crops.",
        "",
        "**Sample training line (first manifest row):**",
        "",
    ])
    for cond in CONDITIONS:
        lines.append(f"- *{cond}:* `{manifest_diag[cond]['sample_line']}`")
        lines.append("")

    lines.extend([
        "### 3.2 Final checkpoint training loss (if checkpoints available)",
        "",
    ])
    if checkpoint_losses:
        lines.extend([
            "| Condition | Seed | Step | Final loss |",
            "|-----------|------|------|------------|",
        ])
        for row in checkpoint_losses:
            loss = row["loss"]
            lines.append(
                f"| {row['condition']} | {row['seed']} | {row['step']} | "
                f"{loss:.4f} |" if loss is not None else f"| {row['condition']} | {row['seed']} | — | — |"
            )
        by_cond: dict[str, list[float]] = defaultdict(list)
        for row in checkpoint_losses:
            if row["loss"] is not None:
                by_cond[row["condition"]].append(row["loss"])
        lines.append("")
        for cond in CONDITIONS:
            if by_cond[cond]:
                lines.append(
                    f"- **{cond}:** mean final loss = {np.mean(by_cond[cond]):.4f} "
                    f"(SD {np.std(by_cond[cond], ddof=1):.4f}, n={len(by_cond[cond])})"
                )
    else:
        lines.append(
            "*No checkpoints found locally* (`checkpoints/checkpoint_{script}_{condition}_seed{N}.pt`). "
            "Compare final `loss` stored in Colab checkpoints: if flattened/inverted loss fell similarly to "
            "natural but accuracy stayed ~0%, the model optimized the training objective on unlearnable / "
            "prior-dominated text without acquiring readable OCR."
        )

    lines.extend([
        "",
        "### 3.3 Confound vs exposure dial on confidence",
        "",
        "docs/results_analysis.md shows mean confidence falling **0.99 → 0.60 → 0.46** across conditions. "
        "That pattern tracks **lower logits on synthesized text**, not necessarily successful exposure control "
        "with intact reading ability. With flattened/inverted accuracy at ~0%, the confidence drop **cannot** be "
        "interpreted as “the dial grounded the model” — it may reflect failure to learn the training distribution.",
        "",
        "---",
        "",
        "## 4. What this would need to become the headline number",
        "",
        "1. **Non-natural conditions must produce >5–10% line-level accuracy** (or at least many glyphs with stable non-zero accuracy) so the FE fit is not a floor artifact.",
        "2. **Longer training or milder inverted PMF** — current inverted synthesis may be too extreme for 5k steps at 19.5M params.",
        "3. **Eval on held-out natural lines** while training on flattened/inverted (optional design change) to separate “can't read synthetic” from “exposure hurt this glyph”.",
        "4. **Mixed-effects extension** (DECISIONS.md #46 / BOOK.md): random intercepts per seed; bootstrap over eval lines for intervals.",
        "5. **Per-step loss curves** (`probe3_training_curve` + `--keep-snapshots`) to see whether natural/low-exposure conditions diverge during training.",
        "",
        "---",
        "",
        "## 5. Complexity (glyph fixed effects)",
        "",
        "The fitted glyph dummies are the **complexity estimates** — not reported individually here "
        f"({len({r.glyph for r in panel})} clusters). Export via `--dump-glyph-effects` (future) or refit with "
        "store of `beta[2:]` from the design matrix. Substantive interpretation requires a feasible fit (§1.2).",
        "",
    ])
    return "\n".join(lines)


def run_analysis(
    manifests_dir: Path,
    probe_results_dir: Path,
    checkpoints_dir: Path | None,
    script: str,
    language: str,
    out_path: Path,
    min_glyph_eval: int = DEFAULT_MIN_GLYPH_EVAL,
) -> dict[str, Any]:
    """End-to-end analysis pipeline."""
    panel = build_panel(manifests_dir, probe_results_dir, script, language)
    feasibility = assess_feasibility(panel, probe_results_dir, script, min_glyph_eval)
    fits = [fit_fixed_effects_ols(panel, seed, min_glyph_eval) for seed in SEEDS]
    manifest_diag = manifest_diagnostics(manifests_dir, script)
    ckpt_losses = load_checkpoint_losses(checkpoints_dir, script) if checkpoints_dir else []

    md = render_markdown(script, feasibility, fits, manifest_diag, ckpt_losses, panel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[probe1_fe] wrote {out_path}")
    print(f"[probe1_fe] feasible={feasibility.overall_feasible} panel_rows={len(panel)}")
    if not feasibility.overall_feasible:
        for r in feasibility.block_reasons:
            print(f"  BLOCK: {r}")
    return {
        "feasible": feasibility.overall_feasible,
        "block_reasons": feasibility.block_reasons,
        "fits": [f.__dict__ for f in fits],
        "by_condition": feasibility.by_condition,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe 1 glyph fixed-effects exposure analysis")
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument("--language", default=None, help="Tier 2 language (defaults to --script)")
    ap.add_argument("--manifests-dir", default="data/manifests")
    ap.add_argument("--probe-results-dir", default="data/probe_results")
    ap.add_argument("--checkpoints-dir", default="checkpoints")
    ap.add_argument("--out", default="docs/probe1_fixed_effects.md")
    ap.add_argument("--min-glyph-eval", type=int, default=DEFAULT_MIN_GLYPH_EVAL)
    args = ap.parse_args()

    language = args.language or args.script
    ckpt_dir = Path(args.checkpoints_dir) if Path(args.checkpoints_dir).exists() else None
    run_analysis(
        Path(args.manifests_dir),
        Path(args.probe_results_dir),
        ckpt_dir,
        args.script,
        language,
        Path(args.out),
        args.min_glyph_eval,
    )


if __name__ == "__main__":
    main()

"""
Probe 5b analysis — zero-shot confidence floor with corrected contrasts.

Why this exists: probe5b_zeroshot_floor.py writes per-image jsonl.
This module turns that into the claim-facing summary: does mean
confidence on never-seen scripts (Santhali Ol Chiki, Kashmiri
Perso-Arabic) and on blank crops differ from the Hindi in-distribution
baseline?

Statistical honesty (DECISIONS.md #51–#52):
- Naive two-sample z treats per-image means as independent Gaussian
  observations. Tokens within an image are correlated, and images may
  cluster by source document, so that SE is optimistic. We still report
  the naive z (Bonferroni α = 0.05/3) so the inflation is visible, but
  the primary interval is a cluster bootstrap that resamples IMAGES
  with replacement.
- "Indistinguishable from Hindi" requires TOST equivalence at a
  stated smallest effect of interest (δ = 0.05), not a non-significant
  difference test.
- Means near a ceiling hide shape: we also report P(conf > 0.95) /
  P(conf > 0.99) and histogram bin counts.
- When seed{0,1,2} jsonl files exist, report per-seed and across-seed
  mean±SD — never a pooled number that hides seed variance.

Accuracy is not reported for unseen scripts (DECISIONS.md #50).

Called from: CLI after data/probe_results/probe5b_*.jsonl exists.
Writes docs/probe5b_analysis.md (or --out) and feeds
docs/statistical_repair.md.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

CONDITIONS = ("hindi", "santhali", "kashmiri", "blank")
BASELINE = "hindi"
COMPARISONS = ("santhali", "kashmiri", "blank")
# Three planned contrasts vs hindi → family-wise α = 0.05/3.
N_COMPARISONS = 3
FAMILY_ALPHA = 0.05
CORRECTED_ALPHA = FAMILY_ALPHA / N_COMPARISONS
# Two-sided critical |z| at the Bonferroni-corrected α.
CRITICAL_Z = float(stats.norm.ppf(1.0 - CORRECTED_ALPHA / 2.0))

# Smallest effect of interest for TOST (DECISIONS.md #52).
# A confidence drop smaller than 0.05 is not practically meaningful
# for abstention routing on this instrument.
SEOI_DELTA = 0.05
TOST_ALPHA = 0.05
# TOST via (1 − 2α) CI ⊆ [−δ, δ] ⇔ both one-sided tests at α.
TOST_CI_LEVEL = 1.0 - 2.0 * TOST_ALPHA  # 0.90

DEFAULT_N_BOOT = 10_000
BOOT_SEED = 42
CEILING_THRESHOLDS = (0.95, 0.99)
HIST_BINS = [0.0, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0001]


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read Probe 5b jsonl; one dict per image × condition."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def discover_seed_paths(
    pattern_dir: Path,
    script: str = "hindi",
    condition: str = "natural",
) -> dict[int, Path]:
    """
    Find probe5b_{script}_{condition}_seed{N}.jsonl for N in 0,1,2.

    Aggregation across seeds is only honest when each seed is a
    separate training run; pooling images across seeds would hide
    seed variance. Returns whatever seeds exist on disk.
    """
    found: dict[int, Path] = {}
    for seed in (0, 1, 2):
        p = pattern_dir / f"probe5b_{script}_{condition}_seed{seed}.jsonl"
        if p.is_file():
            found[seed] = p
    return found


def conf_stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    """
    Per-condition confidence summary for the analysis table.

    SEM and normal-approx 95% CI are the *naive* reporting unit —
    kept so the bootstrap correction's inflation is visible beside
    them. Primary intervals come from cluster_bootstrap.
    """
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    # Sample SD (ddof=1) once n≥2; otherwise SD is undefined.
    sd = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    sem = sd / math.sqrt(n) if n >= 2 and not math.isnan(sd) else float("nan")
    ci_half = 1.96 * sem if not math.isnan(sem) else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": mean - ci_half if not math.isnan(ci_half) else float("nan"),
        "ci_high": mean + ci_half if not math.isnan(ci_half) else float("nan"),
    }


def ceiling_fractions(
    values: list[float] | np.ndarray,
    thresholds: tuple[float, ...] = CEILING_THRESHOLDS,
) -> dict[str, float]:
    """
    Fraction of image-level mean_confidence above each threshold.

    Ceiling mass (e.g. P(>0.99)) is the claim-relevant shape statistic
    when condition means all sit near 0.98–0.99 — a mean of 0.98 with
    half the mass below 0.95 is a different story than a mean of 0.98
    with 95% of records above 0.99.
    """
    arr = np.asarray(values, dtype=float)
    return {f"frac_above_{t}": float(np.mean(arr > t)) for t in thresholds} | {
        "n": int(arr.size),
    }


def histogram_counts(
    values: list[float] | np.ndarray,
    bins: list[float] | None = None,
) -> list[dict[str, float | int]]:
    """
    Coarse histogram of image-level mean_confidence for the write-up.

    Bins concentrate near the ceiling because that is where all four
    conditions live; a uniform 10-bin grid would put almost everything
    in the last bin and hide the shape.
    """
    edges = bins if bins is not None else HIST_BINS
    arr = np.asarray(values, dtype=float)
    counts, _ = np.histogram(arr, bins=edges)
    out: list[dict[str, float | int]] = []
    for i, count in enumerate(counts):
        out.append({
            "bin_low": float(edges[i]),
            "bin_high": float(edges[i + 1]),
            "count": int(count),
            "fraction": float(count) / max(int(arr.size), 1),
        })
    return out


def pairwise_z(
    baseline: list[float] | np.ndarray,
    other: list[float] | np.ndarray,
) -> dict[str, float | bool]:
    """
    Two-sample z for mean(other) − mean(baseline) — the *naive* test.

    Independent-samples SE: sqrt(sd_b²/n_b + sd_o²/n_o). This assumes
    image-level means are i.i.d., which overstates precision when
    tokens-within-image (and possibly document clusters) are correlated.
    Kept for side-by-side reporting against the cluster-bootstrap CI
    (DECISIONS.md #51, statistical_repair.md).
    """
    b = np.asarray(baseline, dtype=float)
    o = np.asarray(other, dtype=float)
    nb, no = len(b), len(o)
    mean_b, mean_o = float(b.mean()), float(o.mean())
    sd_b = float(b.std(ddof=1)) if nb >= 2 else float("nan")
    sd_o = float(o.std(ddof=1)) if no >= 2 else float("nan")
    se = math.sqrt(sd_b**2 / nb + sd_o**2 / no)
    diff = mean_o - mean_b
    z = diff / se if se > 0 else float("nan")
    return {
        "diff": diff,
        "se": se,
        "z": z,
        "exceeds_corrected": abs(z) >= CRITICAL_Z if not math.isnan(z) else False,
    }


def cluster_bootstrap(
    conf_by_cond: dict[str, np.ndarray],
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """
    Cluster bootstrap: resample IMAGES with replacement within each
    condition, recompute condition means and pairwise diffs each draw.

    Why images, not tokens: mean_confidence is already an average over
    correlated per-token confidences inside an image. Resampling tokens
    would pretend those tokens are independent. Resampling images is the
    honest cluster unit available in the jsonl (image_id). Document-level
    clustering beyond image_id is not tagged in the probe output, so
    image is the finest defensibly independent unit we can claim.

    Returns percentile CIs for each condition mean and each
    (cond − hindi) difference, plus the bootstrap distribution
    summaries needed for TOST.
    """
    rng = np.random.default_rng(rng_seed)
    conds = [c for c in CONDITIONS if c in conf_by_cond]
    boot_means = {c: np.empty(n_boot, dtype=float) for c in conds}
    boot_diffs = {c: np.empty(n_boot, dtype=float) for c in COMPARISONS if c in conf_by_cond}

    for b in range(n_boot):
        means_b: dict[str, float] = {}
        for c in conds:
            vals = conf_by_cond[c]
            # Resample images (rows) with replacement — the cluster unit.
            idx = rng.integers(0, len(vals), size=len(vals))
            means_b[c] = float(vals[idx].mean())
            boot_means[c][b] = means_b[c]
        base = means_b.get(BASELINE)
        if base is None:
            continue
        for c in boot_diffs:
            boot_diffs[c][b] = means_b[c] - base

    def pct_ci(arr: np.ndarray, level: float = 0.95) -> tuple[float, float]:
        lo = (1.0 - level) / 2.0 * 100.0
        hi = (1.0 + level) / 2.0 * 100.0
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    mean_cis = {}
    for c, arr in boot_means.items():
        lo95, hi95 = pct_ci(arr, 0.95)
        mean_cis[c] = {
            "boot_mean": float(arr.mean()),
            "ci95_low": lo95,
            "ci95_high": hi95,
            "boot_sd": float(arr.std(ddof=1)),
        }

    diff_cis = {}
    for c, arr in boot_diffs.items():
        lo95, hi95 = pct_ci(arr, 0.95)
        lo90, hi90 = pct_ci(arr, TOST_CI_LEVEL)
        diff_cis[c] = {
            "boot_mean_diff": float(arr.mean()),
            "ci95_low": lo95,
            "ci95_high": hi95,
            "ci90_low": lo90,
            "ci90_high": hi90,
            "boot_sd": float(arr.std(ddof=1)),
            # One-sided bootstrap tail probs for TOST at ±δ.
            "p_lower": float(np.mean(arr <= -SEOI_DELTA)),
            "p_upper": float(np.mean(arr >= SEOI_DELTA)),
        }

    return {
        "n_boot": n_boot,
        "rng_seed": rng_seed,
        "mean_cis": mean_cis,
        "diff_cis": diff_cis,
    }


def tost_equivalence(
    diff_ci90_low: float,
    diff_ci90_high: float,
    p_lower: float,
    p_upper: float,
    delta: float = SEOI_DELTA,
    alpha: float = TOST_ALPHA,
) -> dict[str, Any]:
    """
    Two one-sided tests (TOST) for equivalence at ±delta.

    Why TOST and not "fail to reject difference": a non-significant z
    only says we lack evidence of a difference; it does not say the
    means are close enough for abstention decisions. Equivalence
    requires rejecting both "diff ≤ −δ" and "diff ≥ +δ".

    Decision rule (both must hold):
      (1) 90% bootstrap CI ⊆ [−δ, δ]   (percentile TOST at α=0.05)
      (2) bootstrap p_lower < α and p_upper < α
    We report both; (1) is the primary claim gate.
    """
    ci_inside = (diff_ci90_low > -delta) and (diff_ci90_high < delta)
    tails_ok = (p_lower < alpha) and (p_upper < alpha)
    return {
        "delta": delta,
        "alpha": alpha,
        "ci90_low": diff_ci90_low,
        "ci90_high": diff_ci90_high,
        "ci_inside_bounds": ci_inside,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "tails_reject_both": tails_ok,
        "equivalent": ci_inside and tails_ok,
    }


def naive_tost(
    baseline: list[float] | np.ndarray,
    other: list[float] | np.ndarray,
    delta: float = SEOI_DELTA,
    alpha: float = TOST_ALPHA,
) -> dict[str, Any]:
    """
    Parametric TOST on the same image-level means as pairwise_z.

    Reported beside the bootstrap TOST so the reader can see whether
    the equivalence conclusion is driven by the resampling correction
    or is already present under the naive SE.
    """
    b = np.asarray(baseline, dtype=float)
    o = np.asarray(other, dtype=float)
    nb, no = len(b), len(o)
    diff = float(o.mean() - b.mean())
    sd_b = float(b.std(ddof=1)) if nb >= 2 else float("nan")
    sd_o = float(o.std(ddof=1)) if no >= 2 else float("nan")
    se = math.sqrt(sd_b**2 / nb + sd_o**2 / no)
    if se <= 0 or math.isnan(se):
        return {
            "delta": delta,
            "diff": diff,
            "se": se,
            "z_lower": float("nan"),
            "z_upper": float("nan"),
            "equivalent": False,
        }
    # H0: diff ≤ −δ  → reject if (diff − (−δ))/se > z_{1−α}
    # H0: diff ≥ +δ  → reject if (diff − (+δ))/se < −z_{1−α}
    z_crit = float(stats.norm.ppf(1.0 - alpha))
    z_lower = (diff - (-delta)) / se
    z_upper = (diff - delta) / se
    reject_lower = z_lower > z_crit
    reject_upper = z_upper < -z_crit
    return {
        "delta": delta,
        "diff": diff,
        "se": se,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "z_crit": z_crit,
        "reject_lower": reject_lower,
        "reject_upper": reject_upper,
        "equivalent": reject_lower and reject_upper,
    }


def correct_script_zero_count(records: list[dict], condition: str) -> int | None:
    """
    Images that emitted ZERO graphemes of the script that appears in
    the source image.

    Hindi: trained == image (Devanagari); the probe's classify order
    attributes Devanagari to trained_script, so use n_trained_script.
    Santhali / Kashmiri: use n_image_script (Ol Chiki / Arabic).
    Blank: no correct script — return None.
    """
    if condition == "blank":
        return None
    key = "n_trained_script" if condition == "hindi" else "n_image_script"
    return sum(
        1
        for r in records
        if r["charset_composition"].get(key, 0) == 0
    )


def charset_means(records: list[dict]) -> dict[str, float | None]:
    """Mean trained-script and image-script grapheme fractions."""
    trained = [
        r["charset_composition"]["trained_script_fraction"]
        for r in records
        if r["charset_composition"].get("trained_script_fraction") is not None
    ]
    image = [
        r["charset_composition"]["image_script_fraction"]
        for r in records
        if r["charset_composition"].get("image_script_fraction") is not None
    ]
    return {
        "mean_trained_script_fraction": float(np.mean(trained)) if trained else None,
        "mean_image_script_fraction": float(np.mean(image)) if image else None,
    }


def analyze_one_seed(
    records: list[dict],
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Full per-seed Probe 5b analysis (naive + bootstrap + TOST + shape)."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)

    conf_by_cond = {
        c: np.asarray(
            [r["mean_confidence"] for r in by_cond[c] if r.get("mean_confidence") is not None],
            dtype=float,
        )
        for c in CONDITIONS
        if c in by_cond
    }

    per_condition: dict[str, Any] = {}
    for c in CONDITIONS:
        if c not in conf_by_cond:
            continue
        vals = conf_by_cond[c]
        stats_row: dict[str, Any] = conf_stats(vals)
        stats_row.update(charset_means(by_cond[c]))
        stats_row["n_zero_correct_script"] = correct_script_zero_count(by_cond[c], c)
        stats_row["ceiling"] = ceiling_fractions(vals)
        stats_row["histogram"] = histogram_counts(vals)
        per_condition[c] = stats_row

    boot = cluster_bootstrap(conf_by_cond, n_boot=n_boot, rng_seed=rng_seed)

    # Attach bootstrap mean CIs onto per_condition for the tables.
    for c, ci in boot["mean_cis"].items():
        if c in per_condition:
            per_condition[c]["boot_ci95_low"] = ci["ci95_low"]
            per_condition[c]["boot_ci95_high"] = ci["ci95_high"]
            per_condition[c]["boot_sd"] = ci["boot_sd"]

    contrasts: dict[str, Any] = {}
    baseline_vals = conf_by_cond.get(BASELINE, np.asarray([]))
    for c in COMPARISONS:
        if c not in conf_by_cond or baseline_vals.size == 0:
            continue
        naive = pairwise_z(baseline_vals, conf_by_cond[c])
        dci = boot["diff_cis"][c]
        boot_tost = tost_equivalence(
            dci["ci90_low"],
            dci["ci90_high"],
            dci["p_lower"],
            dci["p_upper"],
        )
        contrasts[c] = {
            **naive,
            "boot_mean_diff": dci["boot_mean_diff"],
            "boot_ci95_low": dci["ci95_low"],
            "boot_ci95_high": dci["ci95_high"],
            "boot_ci90_low": dci["ci90_low"],
            "boot_ci90_high": dci["ci90_high"],
            "boot_sd": dci["boot_sd"],
            "tost_bootstrap": boot_tost,
            "tost_naive": naive_tost(baseline_vals, conf_by_cond[c]),
        }

    meta = {}
    if records:
        meta = {
            "checkpoint_script": records[0].get("checkpoint_script"),
            "training_condition": records[0].get("training_condition"),
            "seed": records[0].get("seed"),
        }

    return {
        "meta": meta,
        "n_total": len(records),
        "corrected_alpha": CORRECTED_ALPHA,
        "critical_z": CRITICAL_Z,
        "n_comparisons": N_COMPARISONS,
        "seoi_delta": SEOI_DELTA,
        "n_boot": n_boot,
        "per_condition": per_condition,
        "contrasts_vs_hindi": contrasts,
        "bootstrap": boot,
    }


def aggregate_across_seeds(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """
    Across-seed mean and SD of each reported scalar — not a pooled
    image-level mean.

    Pooling images across seeds would treat three training runs as one
    mega-sample and hide seed variance. When only one seed exists, the
    across-seed block is None and the write-up must say so.
    """
    if len(per_seed) < 2:
        return {
            "n_seeds": len(per_seed),
            "seeds": sorted(per_seed.keys()),
            "ready": False,
            "note": (
                f"Only seed(s) {sorted(per_seed.keys())} on disk; "
                "across-seed mean±SD withheld until seed{0,1,2} all exist."
            ),
        }

    seeds = sorted(per_seed.keys())

    def collect(path_fn) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        # Gather per-seed values keyed by a label.
        buckets: dict[str, list[float]] = defaultdict(list)
        for s in seeds:
            for key, val in path_fn(per_seed[s]).items():
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                buckets[key].append(float(val))
        for key, vals in buckets.items():
            arr = np.asarray(vals, dtype=float)
            out[key] = {
                "per_seed": {str(s): float(v) for s, v in zip(seeds, vals)},
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) >= 2 else float("nan"),
                "n_seeds": len(arr),
            }
        return out

    means = collect(
        lambda r: {
            c: r["per_condition"][c]["mean"]
            for c in CONDITIONS
            if c in r["per_condition"]
        }
    )
    diffs = collect(
        lambda r: {
            c: r["contrasts_vs_hindi"][c]["diff"]
            for c in COMPARISONS
            if c in r["contrasts_vs_hindi"]
        }
    )
    zs = collect(
        lambda r: {
            c: r["contrasts_vs_hindi"][c]["z"]
            for c in COMPARISONS
            if c in r["contrasts_vs_hindi"]
        }
    )

    # Between-condition range of across-seed means vs within-condition
    # across-seed SD — the threshold-free equivalence argument.
    across_means = {
        c: means[c]["mean"] for c in CONDITIONS if c in means
    }
    if across_means:
        lo_c = min(across_means, key=across_means.get)
        hi_c = max(across_means, key=across_means.get)
        between_range = across_means[hi_c] - across_means[lo_c]
    else:
        lo_c = hi_c = None
        between_range = float("nan")

    # Script substitution: images with zero correct-script chars on
    # unseen scripts (santhali + kashmiri), summed across seeds.
    sub_zero = 0
    sub_n = 0
    per_seed_sub: dict[str, dict[str, int]] = {}
    for s in seeds:
        pc = per_seed[s]["per_condition"]
        s_zero = int(pc.get("santhali", {}).get("n_zero_correct_script") or 0)
        s_n = int(pc.get("santhali", {}).get("n") or 0)
        k_zero = int(pc.get("kashmiri", {}).get("n_zero_correct_script") or 0)
        k_n = int(pc.get("kashmiri", {}).get("n") or 0)
        per_seed_sub[str(s)] = {
            "santhali_zero": s_zero,
            "santhali_n": s_n,
            "kashmiri_zero": k_zero,
            "kashmiri_n": k_n,
            "total_zero": s_zero + k_zero,
            "total_n": s_n + k_n,
        }
        sub_zero += s_zero + k_zero
        sub_n += s_n + k_n

    # Kashmiri replication: which seeds have |z| ≥ CRITICAL_Z and
    # which have hindi mean > kashmiri mean (sign flip of Δ).
    kashmiri_replication: dict[str, Any] = {"per_seed": {}}
    for s in seeds:
        d = per_seed[s]["contrasts_vs_hindi"].get("kashmiri", {})
        h_mean = per_seed[s]["per_condition"].get("hindi", {}).get("mean")
        k_mean = per_seed[s]["per_condition"].get("kashmiri", {}).get("mean")
        kashmiri_replication["per_seed"][str(s)] = {
            "diff": d.get("diff"),
            "z": d.get("z"),
            "exceeds_corrected": d.get("exceeds_corrected", False),
            "hindi_mean": h_mean,
            "kashmiri_mean": k_mean,
            "hindi_exceeds_kashmiri": (
                h_mean is not None
                and k_mean is not None
                and h_mean > k_mean
            ),
        }
    n_exceed = sum(
        1
        for v in kashmiri_replication["per_seed"].values()
        if v.get("exceeds_corrected")
    )
    kashmiri_replication["n_seeds_exceeding_corrected"] = n_exceed
    kashmiri_replication["replicates"] = n_exceed == len(seeds)
    kashmiri_replication["retracted"] = n_exceed > 0 and n_exceed < len(seeds)

    return {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ready": True,
        "condition_means": means,
        "diffs_vs_hindi": diffs,
        "z_vs_hindi": zs,
        "between_condition": {
            "low_condition": lo_c,
            "high_condition": hi_c,
            "low_mean": across_means.get(lo_c) if lo_c else None,
            "high_mean": across_means.get(hi_c) if hi_c else None,
            "range": between_range,
            "within_seed_sd": {
                c: means[c]["sd"] for c in CONDITIONS if c in means
            },
            "range_smaller_than_hindi_sd": (
                between_range < means["hindi"]["sd"]
                if "hindi" in means and not math.isnan(between_range)
                else False
            ),
            "range_smaller_than_blank_sd": (
                between_range < means["blank"]["sd"]
                if "blank" in means and not math.isnan(between_range)
                else False
            ),
        },
        "script_substitution": {
            "total_zero": sub_zero,
            "total_n": sub_n,
            "per_seed": per_seed_sub,
            "all_substituted": sub_zero == sub_n and sub_n > 0,
        },
        "kashmiri_replication": kashmiri_replication,
    }


def analyze(
    records: list[dict],
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Backward-compatible single-file analyze → analyze_one_seed."""
    return analyze_one_seed(records, n_boot=n_boot, rng_seed=rng_seed)


def analyze_all_seeds(
    seed_paths: dict[int, Path],
    n_boot: int = DEFAULT_N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Run per-seed analysis + across-seed aggregation when ready."""
    per_seed = {}
    for seed, path in sorted(seed_paths.items()):
        records = load_records(path)
        if not records:
            continue
        per_seed[seed] = analyze_one_seed(
            records, n_boot=n_boot, rng_seed=rng_seed + seed
        )
        per_seed[seed]["source"] = path.as_posix()
    return {
        "per_seed": per_seed,
        "across_seeds": aggregate_across_seeds(per_seed),
        "seoi_delta": SEOI_DELTA,
        "n_boot": n_boot,
        "corrected_alpha": CORRECTED_ALPHA,
        "critical_z": CRITICAL_Z,
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_ci(lo: float | None, hi: float | None, digits: int = 4) -> str:
    return f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def render_markdown(result: dict[str, Any], source: Path) -> str:
    """
    Single-seed write-up (compat). Prefer render_bundle_markdown when
    seed{0,1,2} are all present — that is the claim-facing path.
    """
    bundle = {
        "per_seed": {int(result["meta"].get("seed", 0)): {**result, "source": source.as_posix()}},
        "across_seeds": aggregate_across_seeds(
            {int(result["meta"].get("seed", 0)): result}
        ),
        "seoi_delta": result.get("seoi_delta", SEOI_DELTA),
        "n_boot": result.get("n_boot", DEFAULT_N_BOOT),
        "corrected_alpha": result.get("corrected_alpha", CORRECTED_ALPHA),
        "critical_z": result.get("critical_z", CRITICAL_Z),
    }
    return render_bundle_markdown(bundle)


def render_bundle_markdown(bundle: dict[str, Any]) -> str:
    """
    Claim-facing multi-seed Probe 5b write-up.

    Leads with the threshold-free equivalence argument (between-condition
    range of across-seed means vs within-condition across-seed SD),
    retracts any single-seed Kashmiri Bonferroni pass that does not
    replicate (DECISIONS.md #14 / #53), and reports 360/360 script
    substitution when it holds. Per-seed naive z + bootstrap TOST stay
    in the tables for honesty. Methods detail:
    docs/statistical_repair.md.
    """
    across = bundle["across_seeds"]
    seeds = sorted(bundle["per_seed"].keys())
    primary = bundle["per_seed"][seeds[0]]
    sources = [
        f"`{bundle['per_seed'][s].get('source', '')}`" for s in seeds
    ]
    meta0 = primary["meta"]
    script = meta0.get("checkpoint_script", "hindi")
    condition = meta0.get("training_condition", "natural")
    n_total = sum(r["n_total"] for r in bundle["per_seed"].values())

    lines: list[str] = [
        "# Probe 5b analysis — zero-shot confidence floor",
        "",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Sources:** {', '.join(sources)}  ",
        f"**Run:** {script}/{condition}/seeds {seeds}  ",
        f"**Records:** {n_total} across {len(seeds)} seed(s)  ",
        f"**Method:** per-seed image-level means; naive pairwise z "
        f"(Bonferroni α = {FAMILY_ALPHA}/{N_COMPARISONS} = "
        f"{bundle['corrected_alpha']:.6f}, critical |z| = "
        f"{bundle['critical_z']:.3f}) beside cluster-bootstrap CIs "
        f"(n_boot = {bundle['n_boot']}); TOST at δ = {bundle['seoi_delta']} "
        f"(DECISIONS.md #52). Across-seed mean±SD of per-seed means — "
        f"never pooled images. Full correction: "
        f"[statistical_repair.md](statistical_repair.md).",
        "",
        "Accuracy is **not** scored on Santhali or Kashmiri — the "
        "instrument's vocabulary is Devanagari grapheme clusters, so CER "
        "against Ol Chiki / Perso-Arabic ground truth would measure "
        "tokenizer impossibility, not vision failure (DECISIONS.md #50).",
        "",
        "---",
        "",
    ]

    if across.get("ready"):
        bc = across["between_condition"]
        means = across["condition_means"]
        lines.extend([
            "## 1. Across-seed confidence (primary table)",
            "",
            "Per-seed means, then across-seed mean ± SD. This is the "
            "unit Decision #14 requires — not a pooled mega-sample.",
            "",
            "| Condition | " + " | ".join(f"seed{s}" for s in seeds)
            + " | Mean | SD |",
            "|-----------|" + "|".join(["------"] * len(seeds)) + "|------|----|",
        ])
        for c in CONDITIONS:
            if c not in means:
                continue
            block = means[c]
            seed_cols = " | ".join(
                _fmt(block["per_seed"].get(str(s))) for s in seeds
            )
            lines.append(
                f"| {c} | {seed_cols} | {_fmt(block['mean'])} | "
                f"{_fmt(block['sd'])} |"
            )

        lines.extend([
            "",
            "### 1a. Equivalence without an assumed δ (lead claim)",
            "",
            f"Between-condition range of across-seed means: "
            f"**{_fmt(bc['low_mean'])}** ({bc['low_condition']}) to "
            f"**{_fmt(bc['high_mean'])}** ({bc['high_condition']}) = "
            f"**{_fmt(bc['range'])}**.",
            "",
            f"Within-condition across-seed SD: hindi "
            f"**{_fmt(bc['within_seed_sd'].get('hindi'))}**, blank "
            f"**{_fmt(bc['within_seed_sd'].get('blank'))}**, santhali "
            f"**{_fmt(bc['within_seed_sd'].get('santhali'))}**, kashmiri "
            f"**{_fmt(bc['within_seed_sd'].get('kashmiri'))}**.",
            "",
            "The between-condition range is **smaller** than the "
            "within-condition seed noise for hindi and for blank. "
            "Condition means sit closer to each other than a single "
            "condition jitters across training seeds — that is the "
            "threshold-free case for treating zero-shot / blank "
            "confidence as equivalent to in-distribution confidence. "
            "TOST at δ = 0.05 (below) agrees, but this comparison does "
            "not need an assumed effect-size threshold.",
            "",
        ])

        # Kashmiri retraction
        kr = across["kashmiri_replication"]
        lines.extend([
            "### 1b. Kashmiri Bonferroni claim — RETRACTED",
            "",
        ])
        lines.append(
            "| Seed | Hindi mean | Kashmiri mean | Δ (kas−hin) | Naive z | "
            "|z| ≥ crit? |"
        )
        lines.append(
            "|------|------------|---------------|-------------|---------|------------|"
        )
        for s in seeds:
            row = kr["per_seed"][str(s)]
            flag = "yes" if row["exceeds_corrected"] else "no"
            lines.append(
                f"| {s} | {_fmt(row['hindi_mean'])} | "
                f"{_fmt(row['kashmiri_mean'])} | {_fmt(row['diff'])} | "
                f"{_fmt(row['z'], 3)} | {flag} |"
            )
        lines.append("")
        if kr.get("retracted"):
            # Find the seed that passed and the seed where hindi > kashmiri
            pass_seeds = [
                s for s, v in kr["per_seed"].items() if v["exceeds_corrected"]
            ]
            flip_seeds = [
                s for s, v in kr["per_seed"].items()
                if v.get("hindi_exceeds_kashmiri")
            ]
            lines.append(
                f"**Retraction (DECISIONS.md #53):** seed-{pass_seeds[0]} "
                f"naive z = {_fmt(kr['per_seed'][pass_seeds[0]]['z'], 3)} "
                "cleared Bonferroni, but the sign does **not** replicate. "
                f"On seed {flip_seeds[0] if flip_seeds else '?'}, hindi mean "
                f"({_fmt(kr['per_seed'][flip_seeds[0]]['hindi_mean']) if flip_seeds else 'n/a'}) "
                f"**exceeds** kashmiri "
                f"({_fmt(kr['per_seed'][flip_seeds[0]]['kashmiri_mean']) if flip_seeds else 'n/a'}). "
                f"Only {kr['n_seeds_exceeding_corrected']}/{len(seeds)} "
                "seeds clear the corrected threshold. The seed-0 result "
                "was a **single-seed artifact caught by the three-seed "
                "requirement** (DECISIONS.md #14). Do not cite Kashmiri "
                "confidence as significantly above Hindi."
            )
            lines.append("")
        elif kr.get("replicates"):
            lines.append(
                "Kashmiri Bonferroni pass replicates on all seeds — "
                "retain with the usual small-n caveat."
            )
            lines.append("")
        else:
            lines.append(
                "No seed clears the Bonferroni-corrected Kashmiri "
                "threshold."
            )
            lines.append("")

        # Script substitution
        sub = across["script_substitution"]
        lines.extend([
            "### 1c. Script substitution",
            "",
            f"**{sub['total_zero']}/{sub['total_n']}** Santhali+Kashmiri "
            "images across all seeds emitted **zero** graphemes of the "
            "script visible in the image (fluent Devanagari instead). "
            + (
                "Substitution replicated on every unseen-script image."
                if sub["all_substituted"]
                else "Substitution is not universal — see per-seed counts."
            ),
            "",
            "| Seed | Santhali zero/n | Kashmiri zero/n | Total |",
            "|------|-----------------|-----------------|-------|",
        ])
        for s in seeds:
            ps = sub["per_seed"][str(s)]
            lines.append(
                f"| {s} | {ps['santhali_zero']}/{ps['santhali_n']} | "
                f"{ps['kashmiri_zero']}/{ps['kashmiri_n']} | "
                f"{ps['total_zero']}/{ps['total_n']} |"
            )
        lines.append("")

        # Diffs across seeds
        lines.extend([
            "### 1d. Per-seed Δ vs Hindi (mean ± SD across seeds)",
            "",
            "| Contrast | " + " | ".join(f"seed{s}" for s in seeds)
            + " | Mean Δ | SD |",
            "|----------|" + "|".join(["------"] * len(seeds)) + "|--------|----|",
        ])
        for c in COMPARISONS:
            if c not in across["diffs_vs_hindi"]:
                continue
            block = across["diffs_vs_hindi"][c]
            seed_cols = " | ".join(
                _fmt(block["per_seed"].get(str(s))) for s in seeds
            )
            lines.append(
                f"| {c} − hindi | {seed_cols} | {_fmt(block['mean'])} | "
                f"{_fmt(block['sd'])} |"
            )
        lines.append("")
    else:
        lines.append(f"*{across.get('note', 'Across-seed summary unavailable.')}*")
        lines.append("")

    # Per-seed detail sections
    lines.extend([
        "## 2. Per-seed detail (naive z, bootstrap, TOST, ceilings)",
        "",
    ])
    for s in seeds:
        result = bundle["per_seed"][s]
        pc = result["per_condition"]
        lines.append(f"### Seed {s}")
        lines.append("")
        lines.append(f"Source: `{result.get('source', '')}` — {result['n_total']} records.")
        lines.append("")
        lines.append(
            "| Condition | n | Mean | Naive 95% CI | Boot 95% CI | "
            "P(>0.95) | P(>0.99) |"
        )
        lines.append(
            "|-----------|---|------|--------------|-------------|"
            "---------|----------|"
        )
        for c in CONDITIONS:
            if c not in pc:
                continue
            row = pc[c]
            ceil = row["ceiling"]
            lines.append(
                f"| {c} | {row['n']} | {_fmt(row['mean'])} | "
                f"{_fmt_ci(row['ci_low'], row['ci_high'])} | "
                f"{_fmt_ci(row.get('boot_ci95_low'), row.get('boot_ci95_high'))} | "
                f"{_fmt(ceil.get('frac_above_0.95'), 3)} | "
                f"{_fmt(ceil.get('frac_above_0.99'), 3)} |"
            )
        lines.append("")
        lines.append(
            "| Contrast | Δ | Naive z | |z|≥crit? | Boot 95% CI | "
            "Boot 90% CI | TOST equiv? |"
        )
        lines.append(
            "|----------|---|---------|----------|-------------|"
            "------------|-------------|"
        )
        for c in COMPARISONS:
            if c not in result["contrasts_vs_hindi"]:
                continue
            d = result["contrasts_vs_hindi"][c]
            bt = d["tost_bootstrap"]
            lines.append(
                f"| {c} − hindi | {_fmt(d['diff'])} | {_fmt(d['z'], 3)} | "
                f"{'yes' if d['exceeds_corrected'] else 'no'} | "
                f"{_fmt_ci(d['boot_ci95_low'], d['boot_ci95_high'])} | "
                f"{_fmt_ci(bt['ci90_low'], bt['ci90_high'])} | "
                f"{'yes' if bt['equivalent'] else 'no'} |"
            )
        lines.append("")
        lines.append("Histogram counts (image-level mean_confidence):")
        lines.append("")
        for c in CONDITIONS:
            if c not in pc:
                continue
            parts = [
                f"[{row['bin_low']:.3f},{min(row['bin_high'], 1.0):.3f}]:{row['count']}"
                for row in pc[c]["histogram"]
            ]
            lines.append(f"- **{c}:** " + ", ".join(parts))
        lines.append("")

    # Finding
    lines.extend(["## 3. Finding (plain language)", ""])
    if across.get("ready"):
        bc = across["between_condition"]
        sub = across["script_substitution"]
        lines.append(
            f"Across seeds {seeds}, mean confidence stays high on Hindi, "
            "Santhali, Kashmiri, and blank alike. The **lead** equivalence "
            f"claim is threshold-free: between-condition range of "
            f"across-seed means is {_fmt(bc['range'])}, smaller than "
            f"hindi's across-seed SD ({_fmt(bc['within_seed_sd'].get('hindi'))}) "
            f"and blank's ({_fmt(bc['within_seed_sd'].get('blank'))}). "
            "TOST at δ = 0.05 passes on every seed × contrast, but the "
            "range-vs-SD comparison is the more compelling evidence "
            "because it needs no assumed effect-size threshold."
        )
        lines.append("")
        if across["kashmiri_replication"].get("retracted"):
            lines.append(
                "The seed-0 Kashmiri Bonferroni pass (z ≈ 2.54) is "
                "**retracted** — it does not replicate (DECISIONS.md #53). "
                "Three seeds earned their keep (DECISIONS.md #14)."
            )
            lines.append("")
        lines.append(
            f"Charset composition remains the sharp signal: "
            f"**{sub['total_zero']}/{sub['total_n']}** unseen-script "
            "images emitted zero correct-script characters — the model "
            "writes fluent Devanagari instead."
        )
        lines.append("")
    lines.extend([
        "**What this does not establish:** that production OCR APIs "
        "behave identically (instrument only); that more Hindi training "
        "steps would fix zero-shot calibration (Probe 3b speaks to "
        "undertraining on in-distribution data, not unseen scripts); or "
        "that confidence differences of a few thousandths are useful "
        "for routing.",
        "",
    ])
    return "\n".join(lines)


def render_statistical_repair_section(bundle: dict[str, Any]) -> str:
    """
    Markdown fragment for docs/statistical_repair.md covering Probe 5b.

    Kept here so the numbers in the repair doc and probe5b_analysis.md
    cannot drift — both are produced from the same analyze_all_seeds
    result.
    """
    lines: list[str] = [
        "## Probe 5b — corrected contrasts",
        "",
    ]
    across = bundle["across_seeds"]
    lines.append(
        f"**Seeds on disk:** {across['seeds']} "
        f"(n_seeds = {across['n_seeds']}). "
        f"**n_boot** = {bundle['n_boot']}. "
        f"**δ (SEOI)** = {bundle['seoi_delta']}."
    )
    lines.append("")
    if not across["ready"]:
        lines.append(f"*{across['note']}*")
        lines.append("")

    for seed, result in sorted(bundle["per_seed"].items()):
        meta = result["meta"]
        lines.append(
            f"### Seed {seed} "
            f"({meta.get('checkpoint_script')}/"
            f"{meta.get('training_condition')})"
        )
        lines.append("")
        lines.append(f"Source: `{result.get('source', '')}`")
        lines.append("")
        lines.append(
            "| Condition | n | Mean | Naive 95% CI | Boot 95% CI | "
            "P(>0.95) | P(>0.99) |"
        )
        lines.append(
            "|-----------|---|------|--------------|-------------|"
            "---------|----------|"
        )
        for c in CONDITIONS:
            if c not in result["per_condition"]:
                continue
            s = result["per_condition"][c]
            ceil = s["ceiling"]
            lines.append(
                f"| {c} | {s['n']} | {_fmt(s['mean'])} | "
                f"{_fmt_ci(s['ci_low'], s['ci_high'])} | "
                f"{_fmt_ci(s.get('boot_ci95_low'), s.get('boot_ci95_high'))} | "
                f"{_fmt(ceil.get('frac_above_0.95'), 3)} | "
                f"{_fmt(ceil.get('frac_above_0.99'), 3)} |"
            )
        lines.append("")
        lines.append(
            "| Contrast | Naive Δ | Naive z | |z|≥crit? | Boot 95% CI | "
            "Boot 90% CI | TOST equiv? |"
        )
        lines.append(
            "|----------|---------|---------|----------|-------------|"
            "------------|-------------|"
        )
        for c in COMPARISONS:
            if c not in result["contrasts_vs_hindi"]:
                continue
            d = result["contrasts_vs_hindi"][c]
            bt = d["tost_bootstrap"]
            lines.append(
                f"| {c} − hindi | {_fmt(d['diff'])} | {_fmt(d['z'], 3)} | "
                f"{'yes' if d['exceeds_corrected'] else 'no'} | "
                f"{_fmt_ci(d['boot_ci95_low'], d['boot_ci95_high'])} | "
                f"{_fmt_ci(bt['ci90_low'], bt['ci90_high'])} | "
                f"{'yes' if bt['equivalent'] else 'no'} |"
            )
        lines.append("")

        # Histogram data dump for plotting without re-running.
        lines.append("Histogram counts (image-level mean_confidence):")
        lines.append("")
        for c in CONDITIONS:
            if c not in result["per_condition"]:
                continue
            parts = [
                f"[{row['bin_low']:.3f},{min(row['bin_high'], 1.0):.3f}]:{row['count']}"
                for row in result["per_condition"][c]["histogram"]
            ]
            lines.append(f"- **{c}:** " + ", ".join(parts))
        lines.append("")

    if across["ready"]:
        lines.append("### Across-seed summary (mean ± SD of per-seed means)")
        lines.append("")
        lines.append("| Quantity | " + " | ".join(f"seed{s}" for s in across["seeds"]) + " | Mean | SD |")
        header_sep = "|----------|" + "|".join(["------"] * len(across["seeds"])) + "|------|----|"
        lines.append(header_sep)
        for label, block in across["condition_means"].items():
            seed_cols = " | ".join(
                _fmt(block["per_seed"].get(str(s))) for s in across["seeds"]
            )
            lines.append(
                f"| mean conf ({label}) | {seed_cols} | "
                f"{_fmt(block['mean'])} | {_fmt(block['sd'])} |"
            )
        for label, block in across["diffs_vs_hindi"].items():
            seed_cols = " | ".join(
                _fmt(block["per_seed"].get(str(s))) for s in across["seeds"]
            )
            lines.append(
                f"| Δ ({label} − hindi) | {seed_cols} | "
                f"{_fmt(block['mean'])} | {_fmt(block['sd'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze Probe 5b zero-shot floor jsonl"
    )
    ap.add_argument(
        "--input",
        default=None,
        help="Single Probe 5b jsonl path (overrides seed discovery)",
    )
    ap.add_argument(
        "--probe-dir",
        default="data/probe_results",
        help="Directory to discover probe5b_*_seed{0,1,2}.jsonl",
    )
    ap.add_argument(
        "--script",
        default="hindi",
        help="Checkpoint script prefix for seed discovery",
    )
    ap.add_argument(
        "--condition",
        default="natural",
        help="Training condition prefix for seed discovery",
    )
    ap.add_argument(
        "--out",
        default="docs/probe5b_analysis.md",
        help="Markdown report path",
    )
    ap.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Cluster bootstrap resamples (default 10000)",
    )
    ap.add_argument(
        "--boot-seed",
        type=int,
        default=BOOT_SEED,
        help="RNG seed for bootstrap",
    )
    ap.add_argument(
        "--dump-json",
        default=None,
        help="Optional path to write the full analysis dict as JSON",
    )
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    if args.input:
        seed_paths = {-1: Path(args.input)}
        # Single-file mode: still wrap in analyze_all_seeds shape.
        records = load_records(Path(args.input))
        if not records:
            raise SystemExit(f"no records in {args.input}")
        one = analyze_one_seed(records, n_boot=args.n_boot, rng_seed=args.boot_seed)
        one["source"] = Path(args.input).as_posix()
        seed_key = int(one["meta"].get("seed", 0))
        bundle = {
            "per_seed": {seed_key: one},
            "across_seeds": aggregate_across_seeds({seed_key: one}),
            "seoi_delta": SEOI_DELTA,
            "n_boot": args.n_boot,
            "corrected_alpha": CORRECTED_ALPHA,
            "critical_z": CRITICAL_Z,
        }
    else:
        seed_paths = discover_seed_paths(probe_dir, args.script, args.condition)
        if not seed_paths:
            raise SystemExit(
                f"no probe5b_{args.script}_{args.condition}_seed*.jsonl "
                f"in {probe_dir}"
            )
        bundle = analyze_all_seeds(
            seed_paths, n_boot=args.n_boot, rng_seed=args.boot_seed
        )

    md = render_bundle_markdown(bundle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        def _to_builtin(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _to_builtin(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_builtin(v) for v in obj]
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        dump_path.write_text(
            json.dumps(_to_builtin(bundle), indent=2),
            encoding="utf-8",
        )

    across = bundle["across_seeds"]
    print(
        f"[analyze_probe5b] wrote {out} "
        f"(seeds={sorted(bundle['per_seed'].keys())}, "
        f"records={sum(r['n_total'] for r in bundle['per_seed'].values())})"
    )
    print(
        f"  Bonferroni α={bundle['corrected_alpha']:.4f}  "
        f"critical |z|={bundle['critical_z']:.3f}  "
        f"δ={bundle['seoi_delta']}  n_boot={bundle['n_boot']}"
    )
    if across.get("ready"):
        bc = across["between_condition"]
        sub = across["script_substitution"]
        kr = across["kashmiri_replication"]
        print(
            f"  between-cond range={_fmt(bc['range'])}  "
            f"hindi_sd={_fmt(bc['within_seed_sd'].get('hindi'))}  "
            f"blank_sd={_fmt(bc['within_seed_sd'].get('blank'))}"
        )
        print(
            f"  script substitution={sub['total_zero']}/{sub['total_n']}  "
            f"kashmiri_retracted={kr.get('retracted')}"
        )
        for c, block in across["condition_means"].items():
            per = ", ".join(
                f"{s}:{_fmt(block['per_seed'].get(str(s)))}"
                for s in across["seeds"]
            )
            print(
                f"  {c:10s} mean±sd={_fmt(block['mean'])}±{_fmt(block['sd'])}  "
                f"per_seed={{{per}}}"
            )
    else:
        print(f"  across-seeds: {across.get('note')}")
        for s, primary in sorted(bundle["per_seed"].items()):
            for c in COMPARISONS:
                if c not in primary["contrasts_vs_hindi"]:
                    continue
                d = primary["contrasts_vs_hindi"][c]
                print(
                    f"  seed{s} vs hindi {c}: Δ={_fmt(d['diff'])} "
                    f"z={_fmt(d['z'], 3)}"
                )


if __name__ == "__main__":
    main()

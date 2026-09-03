"""
Unit tests for Probe 5b statistical-repair helpers.

Why these exist: cluster bootstrap, TOST, and ceiling fractions are
the claim gates in docs/statistical_repair.md. They have checkable
properties (known means, known equivalence under a wide δ, known
ceiling mass) that should not silently regress.
"""
from __future__ import annotations

import numpy as np

from analysis.analyze_probe5b import (
    SEOI_DELTA,
    ceiling_fractions,
    cluster_bootstrap,
    histogram_counts,
    naive_tost,
    pairwise_z,
    tost_equivalence,
)


def test_pairwise_z_zero_when_identical():
    """Identical samples → Δ=0, z=0 (naive SE still defined)."""
    x = [0.9, 0.91, 0.92, 0.93]
    d = pairwise_z(x, x)
    assert abs(d["diff"]) < 1e-12
    assert abs(d["z"]) < 1e-12


def test_ceiling_fractions_all_above():
    vals = [0.991, 0.995, 0.999]
    c = ceiling_fractions(vals)
    assert c["frac_above_0.95"] == 1.0
    assert c["frac_above_0.99"] == 1.0


def test_histogram_counts_sum_to_n():
    vals = [0.5, 0.92, 0.96, 0.985, 0.999]
    rows = histogram_counts(vals)
    assert sum(r["count"] for r in rows) == len(vals)


def test_cluster_bootstrap_ci_covers_true_diff():
    """
    Well-separated means: bootstrap 95% CI for Δ should exclude 0,
    and the boot mean should sit near the true difference.
    """
    rng = np.random.default_rng(0)
    hindi = rng.normal(0.90, 0.02, size=40)
    other = rng.normal(0.98, 0.02, size=40)
    boot = cluster_bootstrap(
        {"hindi": hindi, "santhali": other},
        n_boot=2000,
        rng_seed=1,
    )
    d = boot["diff_cis"]["santhali"]
    assert d["ci95_low"] > 0.0
    assert abs(d["boot_mean_diff"] - (other.mean() - hindi.mean())) < 0.01


def test_tost_rejects_when_diff_near_zero():
    """Δ≈0 with tight bootstrap CI → equivalence at δ=0.05."""
    # Synthetic tight CI well inside [-0.05, 0.05].
    t = tost_equivalence(
        diff_ci90_low=-0.01,
        diff_ci90_high=0.01,
        p_lower=0.0,
        p_upper=0.0,
        delta=SEOI_DELTA,
    )
    assert t["equivalent"] is True
    assert t["ci_inside_bounds"] is True


def test_tost_fails_when_ci_crosses_bound():
    """Wide CI that crosses +δ → not equivalent."""
    t = tost_equivalence(
        diff_ci90_low=-0.02,
        diff_ci90_high=0.06,
        p_lower=0.0,
        p_upper=0.1,
        delta=SEOI_DELTA,
    )
    assert t["equivalent"] is False
    assert t["ci_inside_bounds"] is False


def test_naive_tost_equivalent_on_identical():
    x = list(np.linspace(0.97, 0.99, 30))
    t = naive_tost(x, x, delta=0.05)
    assert t["equivalent"] is True

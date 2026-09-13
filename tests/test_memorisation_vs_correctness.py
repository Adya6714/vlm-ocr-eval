"""Unit tests for the Probe 5 training-overlap split (CPU, no weights)."""
from __future__ import annotations

import math

from analysis.memorisation_vs_correctness import acc_auroc, split_records
from analysis.paper_defensibility_stats import auroc


def test_split_verbatim_only():
    train = {"alpha", "beta"}
    recs = [
        {"ground_truth": "alpha", "confidence": 0.9, "correct": True},
        {"ground_truth": "gamma", "confidence": 0.1, "correct": False},
        {"ground_truth": "beta", "confidence": 0.8, "correct": False},
    ]
    matching, nonmatching = split_records(recs, train)
    assert [r["ground_truth"] for r in matching] == ["alpha", "beta"]
    assert [r["ground_truth"] for r in nonmatching] == ["gamma"]


def test_auroc_empty_nonmatching_is_nan():
    rows = [
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.1, "correct": False},
    ]
    n, acc, auc = acc_auroc(rows)
    assert n == 2
    assert acc == 0.5
    assert auc == auroc([0.9, 0.1], [1, 0])
    n0, acc0, auc0 = acc_auroc([])
    assert n0 == 0
    assert math.isnan(acc0) and math.isnan(auc0)

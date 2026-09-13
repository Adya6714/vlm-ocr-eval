"""
Checkable properties of Stage 6 residual CER and matched-k routing.
"""

from __future__ import annotations

import numpy as np

from probes.cascade import (
    escalate_highest_scores,
    escalate_lowest_scores,
    residual_system_cer,
)


def test_residual_perfect_escalation_of_all_errors():
    """If the two error pages are escalated, leftover system CER is 0."""
    cers = [0.0, 0.0, 1.0, 0.5]
    mask = np.array([False, False, True, True])
    assert residual_system_cer(cers, mask) == 0.0


def test_residual_escalating_clean_pages_leaves_errors():
    """Escalating the already-correct pages does not help."""
    cers = [0.0, 0.0, 1.0, 0.5]
    mask = np.array([True, True, False, False])
    assert abs(residual_system_cer(cers, mask) - 0.375) < 1e-12


def test_escalate_lowest_picks_the_small_scores():
    scores = [0.9, 0.1, 0.5]
    mask = escalate_lowest_scores(scores, 1)
    assert mask.tolist() == [False, True, False]


def test_escalate_highest_picks_the_large_scores():
    scores = [3.0, 10.0, 1.0]
    mask = escalate_highest_scores(scores, 1)
    assert mask.tolist() == [False, True, False]

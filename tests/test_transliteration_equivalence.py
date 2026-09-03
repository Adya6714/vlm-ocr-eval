"""
Tier 2 validation set must pass end-to-end through tier2_equivalent.

Why: a stub set of only negatives makes a 0% corpus Tier-2 rate look
like "not implemented." These tests lock the honest ~38-pair set
(DECISIONS.md #54) so a future edit cannot silently empty the positives.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))

from transliteration_equivalence import (  # noqa: E402
    SCRIPT_MAP,
    VALIDATION_SET,
    tier2_equivalent,
)


def test_validation_set_size_and_ratio():
    """At least ~25 scored pairs; positives outnumber different-word negatives."""
    scored = [p for p in VALIDATION_SET if p[2] in SCRIPT_MAP]
    assert len(scored) >= 25
    n_pos = sum(1 for *_, exp in scored if exp)
    n_neg = sum(1 for *_, exp in scored if not exp)
    assert n_pos >= 15
    assert n_neg >= 8
    # Roughly 1:2 negatives:positives (allow slack for boundary rows).
    assert n_neg <= n_pos


def test_validation_set_all_pass():
    """Every hand-picked pair matches expected_equivalent."""
    failures = []
    for ref, hyp, language, expected in VALIDATION_SET:
        if language not in SCRIPT_MAP:
            continue
        actual = tier2_equivalent(ref, hyp, language)
        if actual != expected:
            failures.append((ref, hyp, language, expected, actual))
    assert failures == [], failures


def test_short_long_vowel_not_collapsed():
    """Decision #18: short/long vowel pairs must remain non-equivalent."""
    assert tier2_equivalent("कमल", "कमाल", "hindi") is False
    assert tier2_equivalent("दिन", "दीन", "hindi") is False


def test_om_symbol_vs_o_anusvara():
    """Tier-2-only residual: ॐ and ओं share ISO ōṁ; Tier 1 does not equate."""
    assert tier2_equivalent("ॐ", "ओं", "hindi") is True

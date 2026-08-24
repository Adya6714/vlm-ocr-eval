"""Checkable properties of the glyph-frequency dial."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renderer.glyph_frequency import (  # noqa: E402
    TARGET_TV_TOLERANCE,
    count_clusters,
    grapheme_clusters,
    matches_target,
    normalize_counts,
    resample_corpus,
    target_counts_from_pmf,
    target_distribution,
    total_variation,
)


def _toy_corpus():
    # अ appears a lot, ज्ञ almost never — classic Probe 1 contrast.
    common = ["अमा अमा अमा काला "] * 20
    rare = ["ज्ञानी ज्ञ "] * 2
    mid = ["राम सीता किताब "] * 8
    return common + rare + mid


def _hindi_corpus():
    path = ROOT / "data" / "raw" / "hindi" / "ground_truth.jsonl"
    if not path.exists():
        return None
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]


class GraphemeTests(unittest.TestCase):
    def test_conjunct_is_one_cluster(self):
        clusters = grapheme_clusters("क्ष")
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0], "क्ष")

    def test_matra_joins_base(self):
        clusters = grapheme_clusters("का")
        self.assertEqual(clusters, ["का"])


class TargetDistributionTests(unittest.TestCase):
    def test_natural_is_identity(self):
        counts = count_clusters(_toy_corpus())
        natural = normalize_counts(counts)
        target = target_distribution(counts, "natural")
        self.assertLess(total_variation(natural, target), 1e-12)

    def test_flattened_is_uniform(self):
        counts = count_clusters(_toy_corpus())
        flat = target_distribution(counts, "flattened")
        n = len(flat)
        for v in flat.values():
            self.assertAlmostEqual(v, 1.0 / n)

    def test_inverted_swaps_ranks(self):
        counts = count_clusters(_toy_corpus())
        natural = normalize_counts(counts)
        inv = target_distribution(counts, "inverted")
        rarest = min(natural, key=natural.get)
        commonest = max(natural, key=natural.get)
        self.assertGreater(inv[rarest], inv[commonest])


class QuotaTests(unittest.TestCase):
    def test_target_counts_sum_and_near_pmf(self):
        counts = count_clusters(_toy_corpus())
        target = target_distribution(counts, "flattened")
        total = sum(counts.values())
        quota = target_counts_from_pmf(target, total)
        self.assertEqual(sum(quota.values()), total)
        realized = normalize_counts(quota)
        self.assertLess(total_variation(realized, target), 0.02)


class ResampleTests(unittest.TestCase):
    def test_natural_tv_near_zero(self):
        result = resample_corpus(_toy_corpus(), "natural", rng=np.random.default_rng(0))
        self.assertLess(result.tv_distance, 0.01)
        self.assertTrue(result.within_tolerance())

    def test_toy_flattened_within_tolerance(self):
        result = resample_corpus(_toy_corpus(), "flattened", rng=np.random.default_rng(1))
        self.assertTrue(
            matches_target(result.realized, result.target),
            msg=f"TV={result.tv_distance}",
        )
        self.assertLessEqual(result.tv_distance, TARGET_TV_TOLERANCE)

    def test_toy_inverted_within_tolerance_and_promotes_rare(self):
        texts = _toy_corpus()
        natural = normalize_counts(count_clusters(texts))
        rarest = min(natural, key=natural.get)
        result = resample_corpus(texts, "inverted", rng=np.random.default_rng(2))
        self.assertLessEqual(result.tv_distance, TARGET_TV_TOLERANCE)
        self.assertGreater(result.realized.get(rarest, 0.0), natural[rarest])


@unittest.skipUnless(_hindi_corpus() is not None, "Hindi GlotOCR ground truth missing")
class HindiCorpusAcceptanceTests(unittest.TestCase):
    """
    The audit gate: on the real 60-line Hindi slice, flattened and
    inverted must clear TARGET_TV_TOLERANCE. Sentence-IS alone could not
    (TV 0.3–0.65); synthesis must.
    """

    def test_all_modes_within_tolerance(self):
        texts = _hindi_corpus()
        for mode in ("natural", "flattened", "inverted"):
            result = resample_corpus(texts, mode, rng=np.random.default_rng(0))
            self.assertLessEqual(
                result.tv_distance,
                TARGET_TV_TOLERANCE,
                msg=f"{mode} TV={result.tv_distance} > {TARGET_TV_TOLERANCE}",
            )

    def test_inverted_raises_rare_quartile_mass(self):
        texts = _hindi_corpus()
        natural = normalize_counts(count_clusters(texts))
        ranked = sorted(natural, key=natural.get)
        rare_q = set(ranked[: max(1, len(ranked) // 4)])
        nat_rare = sum(natural[c] for c in rare_q)
        inv = resample_corpus(texts, "inverted", rng=np.random.default_rng(0))
        inv_rare = sum(inv.realized.get(c, 0.0) for c in rare_q)
        self.assertGreater(inv_rare, nat_rare)


if __name__ == "__main__":
    unittest.main()

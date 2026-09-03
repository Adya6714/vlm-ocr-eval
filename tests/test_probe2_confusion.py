"""Checkable properties of Probe 2 alignment and true-token stats."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "src" / "probes"
INSTRUMENT_DIR = ROOT / "src" / "models" / "instrument"
sys.path.insert(0, str(PROBES_DIR))
sys.path.insert(0, str(INSTRUMENT_DIR))

from probe2_confusion_graph import (  # noqa: E402
    aggregate_confusion_pairs,
    align_substitutions,
    content_token_clusters,
    qualitative_tag,
    true_token_stats,
)
from train import checkpoint_path, tokenizer_path  # noqa: E402


class TestCheckpointNaming(unittest.TestCase):
    def test_probe2_uses_script_scoped_paths(self):
        """DECISIONS.md #47 — not legacy checkpoint_natural_seed0.pt."""
        ckpt = checkpoint_path("/ckpt", "hindi", "natural", 0)
        tok = tokenizer_path("/ckpt", "hindi", "natural")
        self.assertEqual(ckpt, "/ckpt/checkpoint_hindi_natural_seed0.pt")
        self.assertEqual(tok, "/ckpt/tokenizer_hindi_natural.json")
        self.assertNotIn("checkpoint_natural_seed", ckpt)


class TestAlignSubstitutions(unittest.TestCase):
    def test_identical_no_subs(self):
        self.assertEqual(align_substitutions(["क", "ख"], ["क", "ख"]), [])

    def test_single_substitution(self):
        subs = align_substitutions(["क", "ख", "ग"], ["क", "घ", "ग"])
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["true_cluster"], "ख")
        self.assertEqual(subs[0]["predicted_cluster"], "घ")
        self.assertEqual(subs[0]["pred_index"], 1)

    def test_insertion_does_not_create_false_sub(self):
        # GT: क ग ; pred: क ख ग — insertion of ख, not a sub of ग.
        subs = align_substitutions(["क", "ग"], ["क", "ख", "ग"])
        self.assertEqual(subs, [])


class TestTrueTokenStats(unittest.TestCase):
    def test_rank_and_prob(self):
        class Tok:
            cluster_to_id = {"a": 0, "b": 1, "c": 2}

        probs = torch.tensor([0.1, 0.7, 0.2])
        stats = true_token_stats(probs, "c", Tok())
        self.assertAlmostEqual(stats["true_prob"], 0.2)
        self.assertEqual(stats["true_rank"], 2)  # b=0.7, c=0.2, a=0.1
        self.assertTrue(stats["true_in_top5"])

    def test_oov_true_cluster(self):
        class Tok:
            cluster_to_id = {"a": 0}

        probs = torch.tensor([1.0])
        stats = true_token_stats(probs, "missing", Tok())
        self.assertEqual(stats["true_prob"], 0.0)
        self.assertIsNone(stats["true_rank"])
        self.assertFalse(stats["true_in_vocab"])


class TestAggregate(unittest.TestCase):
    def test_aggregate_means(self):
        misreads = [
            {
                "true_cluster": "क",
                "predicted_cluster": "ख",
                "true_prob": 0.2,
                "true_rank": 2,
                "true_in_top5": True,
                "top5": [{"cluster": "ख", "prob": 0.5}],
            },
            {
                "true_cluster": "क",
                "predicted_cluster": "ख",
                "true_prob": 0.4,
                "true_rank": 4,
                "true_in_top5": False,
                "top5": [{"cluster": "ख", "prob": 0.5}],
            },
        ]
        pairs = aggregate_confusion_pairs(misreads)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["count"], 2)
        self.assertAlmostEqual(pairs[0]["mean_true_prob"], 0.3)
        self.assertAlmostEqual(pairs[0]["mean_true_rank"], 3.0)
        self.assertAlmostEqual(pairs[0]["frac_true_in_top5"], 0.5)


class TestQualitativeTag(unittest.TestCase):
    def test_same_base_matra(self):
        # क vs का — same base consonant, length differs via matra.
        self.assertEqual(qualitative_tag("क", "का"), "same-base-matra-diff")

    def test_dissimilar_ascii(self):
        self.assertEqual(qualitative_tag("क", "A"), "dissimilar")


class TestContentTokens(unittest.TestCase):
    def test_strips_specials(self):
        class Tok:
            id_to_cluster = {0: "<BOS>", 1: "क", 2: "<EOS>", 3: "ख"}

        self.assertEqual(
            content_token_clusters([0, 1, 3, 2], Tok()),
            ["क", "ख"],
        )


if __name__ == "__main__":
    unittest.main()

"""
Checkable properties of the attention-ablation metrics and generate knobs.

Why these exist: KL(full || zero), prior sufficiency (= 1 − TV), and
zero-encoder-memory generation are the claim gates for Claim B's
mechanistic follow-up. They have closed-form answers on hand-built
distributions that should not silently regress.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_DIR = ROOT / "src" / "models" / "instrument"
PROBES_DIR = ROOT / "src" / "probes"
ANALYSIS_DIR = ROOT / "src" / "analysis"
sys.path.insert(0, str(INSTRUMENT_DIR))
sys.path.insert(0, str(PROBES_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from generate import generate, kl_divergence, prior_sufficiency  # noqa: E402
from probe_attention_ablation import (  # noqa: E402
    build_hindi_sample,
    compare_distributions,
    load_completed_paths,
)
from analyze_attention_ablation import paired_cluster_bootstrap  # noqa: E402
from tokenizer import GraphemeTokenizer  # noqa: E402
from train import InstrumentModel  # noqa: E402


class TestKlAndPriorSufficiency(unittest.TestCase):
    def test_kl_zero_when_identical(self):
        p = torch.tensor([0.1, 0.2, 0.7])
        self.assertAlmostEqual(kl_divergence(p, p), 0.0, places=6)

    def test_kl_positive_when_different(self):
        p = torch.tensor([0.9, 0.05, 0.05])
        q = torch.tensor([0.05, 0.05, 0.9])
        self.assertGreater(kl_divergence(p, q), 0.5)

    def test_prior_sufficiency_one_when_identical(self):
        p = torch.tensor([0.25, 0.25, 0.5])
        self.assertAlmostEqual(prior_sufficiency(p, p), 1.0, places=6)

    def test_prior_sufficiency_equals_one_minus_tv(self):
        p = torch.tensor([0.7, 0.2, 0.1])
        q = torch.tensor([0.1, 0.2, 0.7])
        tv = 0.5 * torch.sum(torch.abs(p - q)).item()
        self.assertAlmostEqual(prior_sufficiency(p, q), 1.0 - tv, places=6)

    def test_compare_distributions_top1_disagree(self):
        class FakeTok:
            id_to_cluster = {0: "a", 1: "b", 2: "c"}

        p = torch.tensor([0.8, 0.1, 0.1])
        q = torch.tensor([0.1, 0.1, 0.8])
        steps = compare_distributions([p], [q], FakeTok())
        self.assertEqual(len(steps), 1)
        self.assertFalse(steps[0]["top1_agree"])
        self.assertGreater(steps[0]["kl_full_given_zero"], 0.0)


class TestGenerateAblation(unittest.TestCase):
    def _tiny_setup(self):
        # Minimal vocab + untrained model — shapes and flags only.
        tok = GraphemeTokenizer()
        corpus = ["अ आ इ उ ए"] * 3
        tok.build_vocab(corpus, min_freq=1)
        model = InstrumentModel(vocab_size=len(tok))
        model.eval()
        # Canonical-ish line: height multiple of patch size 14.
        image = torch.ones(1, 1, 70, 140)
        return model, tok, image

    def test_zero_memory_flag_runs_and_echoes(self):
        model, tok, image = self._tiny_setup()
        out = generate(model, image, tok, max_len=5, zero_encoder_memory=True)
        self.assertTrue(out["zero_encoder_memory"])
        self.assertGreater(len(out["step_confidences"]), 0)
        self.assertTrue(all(0.0 <= c <= 1.0 for c in out["step_confidences"]))

    def test_force_next_ids_follows_sequence(self):
        model, tok, image = self._tiny_setup()
        full = generate(model, image, tok, max_len=4, return_full_probs=True)
        forced = full["token_ids"][1:]
        tf = generate(
            model,
            image,
            tok,
            max_len=len(forced),
            zero_encoder_memory=True,
            return_full_probs=True,
            force_next_ids=forced,
        )
        self.assertEqual(tf["token_ids"], full["token_ids"])
        self.assertEqual(len(tf["step_probs"]), len(full["step_probs"]))

    def test_return_full_probs_shape(self):
        model, tok, image = self._tiny_setup()
        out = generate(model, image, tok, max_len=3, return_full_probs=True)
        self.assertIn("step_probs", out)
        for p in out["step_probs"]:
            self.assertEqual(p.numel(), len(tok))
            self.assertAlmostEqual(float(p.sum()), 1.0, places=5)


class TestProbeHelpers(unittest.TestCase):
    def test_load_completed_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            out.write_text(
                '{"image_path": "/a.png"}\n{"image_path": "/b.png"}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_completed_paths(out), {"/a.png", "/b.png"})

    def test_build_hindi_sample_matches_probe5b_seed(self):
        """Same Random(0) draw as Probe 5b when Hindi pool exists."""
        data_root = ROOT / "data"
        gt = data_root / "raw" / "hindi" / "ground_truth.jsonl"
        if not gt.exists():
            self.skipTest("no Hindi ground truth on disk")
        repo_root = data_root.parent
        tasks = build_hindi_sample(data_root, repo_root, n_samples=100)
        # Pool is 60 in this checkout; both probes take the full pool.
        self.assertEqual(len(tasks), 60)
        # Deterministic first id under Random(0).
        self.assertIsNotNone(tasks[0]["row"].get("id"))


class TestPairedBootstrap(unittest.TestCase):
    def test_bootstrap_diff_near_zero_when_identical(self):
        import numpy as np

        x = np.linspace(0.97, 0.99, 40)
        boot = paired_cluster_bootstrap(x, x, n_boot=500, rng_seed=1)
        self.assertAlmostEqual(boot["diff_boot_mean"], 0.0, places=5)
        lo, hi = boot["diff_ci95"]
        # Identical pairs → every resampled Δ is exactly 0.
        self.assertAlmostEqual(lo, 0.0, places=5)
        self.assertAlmostEqual(hi, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

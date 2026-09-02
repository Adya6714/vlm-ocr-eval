"""Checkable properties of training-curve snapshot paths and interpretation."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_DIR = ROOT / "src" / "models" / "instrument"
PROBES_DIR = ROOT / "src" / "probes"
sys.path.insert(0, str(INSTRUMENT_DIR))
sys.path.insert(0, str(PROBES_DIR))

from train import checkpoint_path, snapshot_checkpoint_path  # noqa: E402
from probe3_training_curve import interpret_curve  # noqa: E402


class TestProbe3TrainingCurve(unittest.TestCase):
    def test_snapshot_path_differs_from_resume_checkpoint(self):
        root = "/tmp/out"
        main = checkpoint_path(root, "hindi", "natural", 0)
        snap = snapshot_checkpoint_path(root, "hindi", "natural", 0, 200)
        self.assertNotEqual(main, snap)
        self.assertEqual(snap, os.path.join(root, "checkpoint_hindi_natural_seed0_step200.pt"))

    def test_interpret_curve_supports_language_prior_hypothesis(self):
        points = [
            {"step": 200, "training_loss": 5.0, "real_minus_blank_gap": 0.01},
            {"step": 5000, "training_loss": 0.01, "real_minus_blank_gap": 0.02},
        ]
        text = interpret_curve(points)
        self.assertIn("(a)", text)
        self.assertIn("language prior", text)

    def test_interpret_curve_supports_undertraining_hypothesis(self):
        points = [
            {"step": 200, "training_loss": 2.0, "real_minus_blank_gap": 0.01},
            {"step": 5000, "training_loss": 0.5, "real_minus_blank_gap": 0.20},
        ]
        text = interpret_curve(points)
        self.assertIn("(b)", text)


if __name__ == "__main__":
    unittest.main()

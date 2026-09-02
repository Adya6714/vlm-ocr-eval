"""Checkable properties for Probe 1 fixed-effects pre-checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "renderer"))

from probe1_fixed_effects import (  # noqa: E402
    _align_glyph_matches,
    assess_feasibility,
    build_panel,
    load_probe5_line_accuracy,
    load_training_exposure,
    per_glyph_outcomes,
)


class TestProbe1FixedEffects(unittest.TestCase):
    def test_training_exposure_counts_glyph_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "hindi_natural.jsonl"
            manifest.write_text(
                json.dumps({"image_path": "x.png", "text": "हह"}) + "\n",
                encoding="utf-8",
            )
            exp = load_training_exposure(manifest)
            self.assertGreaterEqual(exp.get("ह", 0), 1)

    def test_per_glyph_all_correct_when_line_correct(self):
        pairs = per_glyph_outcomes("हिन्दी", "हिन्दी", "hindi", line_correct=True)
        self.assertTrue(all(ok for _, ok in pairs))

    def test_align_finds_exact_cluster_matches(self):
        matched = _align_glyph_matches(["क", "ा"], ["क", "ा"])
        self.assertEqual(matched, {0, 1})

    def test_probe5_line_accuracy_from_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            path = d / "probe5_hindi_natural_seed0.jsonl"
            path.write_text(
                json.dumps({
                    "records": [
                        {"confidence": 0.9, "correct": True, "prediction": "a", "ground_truth": "a"},
                        {"confidence": 0.9, "correct": False, "prediction": "b", "ground_truth": "c"},
                    ],
                    "buckets": [],
                }),
                encoding="utf-8",
            )
            stats = load_probe5_line_accuracy(d, "hindi", "natural", seeds=(0,))
            self.assertEqual(stats["n_lines"], 2)
            self.assertAlmostEqual(stats["line_accuracy"], 0.5)

    def test_feasibility_blocks_near_zero_flattened_inverted(self):
        if not Path("data/probe_results/probe5_hindi_natural_seed0.jsonl").exists():
            self.skipTest("probe results not on disk")
        panel = build_panel(
            Path("data/manifests"), Path("data/probe_results"), "hindi", "hindi",
        )
        report = assess_feasibility(panel, Path("data/probe_results"), "hindi")
        self.assertFalse(report.overall_feasible)
        self.assertTrue(any("below 2%" in r or "below 5%" in r for r in report.block_reasons))


if __name__ == "__main__":
    unittest.main()

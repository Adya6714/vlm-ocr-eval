"""RLVR reward terms and pairwise orderer — checkable without a GPU."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models" / "demo"))
sys.path.insert(0, str(ROOT / "src" / "eval"))

from reading_order_module import count_sort_order  # noqa: E402
from layout_module import blocks_from_page_gt  # noqa: E402
from rlvr import ablation_omission_signal, char_accuracy, compute_reward, coverage  # noqa: E402


class PairwiseOrderTests(unittest.TestCase):
    def test_two_column_ltr_within_band(self):
        blocks = [
            {"id": "R", "x": 0.55, "y": 0.10},
            {"id": "L", "x": 0.10, "y": 0.11},
            {"id": "B", "x": 0.10, "y": 0.50},
        ]
        # L and R are in the same band (y_tol=0.02) so L before R, then B.
        self.assertEqual(count_sort_order(blocks), ["L", "R", "B"])


class LayoutOracleTests(unittest.TestCase):
    def test_blocks_from_lines(self):
        gt = {
            "width": 100,
            "height": 200,
            "lines": [
                {
                    "bbox": [10, 20, 90, 40],
                    "region": "body",
                    "reading_order": 0,
                    "text": "a",
                },
                {
                    "bbox": [10, 50, 90, 70],
                    "region": "body",
                    "reading_order": 1,
                    "text": "b",
                },
            ],
        }
        blocks = blocks_from_page_gt(gt)
        self.assertEqual([b["id"] for b in blocks], ["body_0", "body_1"])
        self.assertAlmostEqual(blocks[0]["y"], 0.1)


class RewardTests(unittest.TestCase):
    def test_perfect_copy(self):
        r = compute_reward("भारत", "भारत", lambda_coverage=1.0)
        self.assertAlmostEqual(r["char_acc"], 1.0)
        self.assertAlmostEqual(r["coverage"], 1.0)
        self.assertAlmostEqual(r["reward"], 1.0)

    def test_empty_hyp_has_zero_coverage(self):
        r = compute_reward("", "abcdefghij", lambda_coverage=1.0)
        self.assertEqual(r["coverage"], 0.0)
        self.assertLess(r["reward"], 0.0)

    def test_ablation_omit_beats_full_without_coverage(self):
        gt = "easy sentence. rareglyphcluster hard tail."
        easy = "easy sentence."
        sig = ablation_omission_signal(gt, easy)
        self.assertTrue(sig["omit_beats_full_without_coverage"])
        self.assertTrue(sig["coverage_term_penalizes_omit"])
        self.assertLess(sig["omit"]["coverage"], 1.0)

    def test_char_accuracy_penalizes_deletions_against_gt(self):
        # Without emitted-only, omitting the tail is *not* 1.0 accuracy.
        acc = char_accuracy("abc", "abcdef")
        self.assertLess(acc, 1.0)
        self.assertGreater(coverage("abc", "abcdef"), 0.0)


if __name__ == "__main__":
    unittest.main()

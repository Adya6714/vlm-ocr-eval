"""Checkable properties of Kendall-tau reading order and table binding."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))

from reading_order_metric import (  # noqa: E402
    geometric_pred_order,
    inventory_layout_bank,
    kendall_tau_order,
    tau_by_bucket,
)
from table_binding import CellBox, HeaderBox, bind_cell_to_header, binding_accuracy  # noqa: E402


class KendallTauTests(unittest.TestCase):
    def test_identity_is_one(self):
        ids = ["a", "b", "c", "d"]
        self.assertAlmostEqual(kendall_tau_order(ids, ids), 1.0)

    def test_reversal_is_minus_one(self):
        ids = ["a", "b", "c", "d"]
        self.assertAlmostEqual(kendall_tau_order(ids, list(reversed(ids))), -1.0)

    def test_adjacent_swap_better_than_reversal(self):
        ids = ["a", "b", "c", "d"]
        swapped = ["a", "c", "b", "d"]
        tau_swap = kendall_tau_order(ids, swapped)
        tau_rev = kendall_tau_order(ids, list(reversed(ids)))
        self.assertGreater(tau_swap, tau_rev)
        self.assertGreater(tau_swap, 0.0)

    def test_rejects_missing_block(self):
        with self.assertRaises(ValueError):
            kendall_tau_order(["a", "b", "c"], ["a", "b"])

    def test_singleton_is_one(self):
        self.assertEqual(kendall_tau_order(["only"], ["only"]), 1.0)

    def test_bucket_curve_omits_empty_as_none(self):
        pages = [
            {
                "category": "single-column",
                "gt_order": ["t", "b"],
                "pred_order": ["t", "b"],
            },
            {
                "category": "two-column",
                "gt_order": ["L", "R"],
                "pred_order": ["R", "L"],
            },
        ]
        curve = tau_by_bucket(pages)
        self.assertEqual(curve["single-column"]["n"], 1)
        self.assertAlmostEqual(curve["single-column"]["mean_tau"], 1.0)
        self.assertAlmostEqual(curve["two-column"]["mean_tau"], -1.0)
        self.assertEqual(curve["form"]["n"], 0)
        self.assertIsNone(curve["form"]["mean_tau"])
        self.assertEqual(curve["table-embedded"]["n"], 0)

    def test_geometric_ltr_sort(self):
        blocks = [
            {"id": "right", "x": 0.6, "y": 0.1},
            {"id": "left", "x": 0.1, "y": 0.1},
            {"id": "below", "x": 0.1, "y": 0.5},
        ]
        self.assertEqual(geometric_pred_order(blocks), ["left", "right", "below"])


class TableBindingTests(unittest.TestCase):
    def setUp(self):
        self.headers = [
            HeaderBox(0, "name", 0.0, 0.0, 0.3, 0.1),
            HeaderBox(1, "age", 0.3, 0.0, 0.6, 0.1),
            HeaderBox(2, "city", 0.6, 0.0, 1.0, 0.1),
        ]
        self.cells = [
            CellBox(0, 0, "Priya", 0.02, 0.15, 0.28, 0.25),
            CellBox(0, 1, "27", 0.32, 0.15, 0.58, 0.25),
            CellBox(0, 2, "Mumbai", 0.62, 0.15, 0.98, 0.25),
        ]

    def test_geometry_perfect_on_aligned_grid(self):
        acc = binding_accuracy(self.cells, self.headers)
        self.assertEqual(acc["n"], 3)
        self.assertEqual(acc["correct"], 3)
        self.assertEqual(acc["accuracy"], 1.0)

    def test_shifted_cell_still_binds_if_overlap_wins(self):
        shifted = CellBox(0, 0, "Priya", 0.05, 0.15, 0.35, 0.25)
        self.assertEqual(bind_cell_to_header(shifted, self.headers), 0)

    def test_pred_cols_can_be_wrong(self):
        acc = binding_accuracy(self.cells, self.headers, pred_cols=[2, 1, 0])
        self.assertEqual(acc["correct"], 1)
        self.assertAlmostEqual(acc["accuracy"], 1 / 3)

    def test_empty_table(self):
        acc = binding_accuracy([], self.headers)
        self.assertEqual(acc["n"], 0)
        self.assertIsNone(acc["accuracy"])


class LayoutBankInventoryTests(unittest.TestCase):
    def test_bank_is_partial_and_scorable(self):
        bank = ROOT / "data" / "cache" / "layouts" / "bank.json"
        inv = inventory_layout_bank(bank)
        self.assertGreaterEqual(inv["n_templates"], 1)
        self.assertEqual(inv["category_counts"].get("form", 0), 0)
        self.assertEqual(inv["n_real_table_regions"], 0)
        self.assertEqual(inv["n_form_field_regions"], 0)
        self.assertGreater(inv["category_counts"]["single-column"], 0)
        sc = inv["geometric_baseline_tau"]["single-column"]
        self.assertGreater(sc["n"], 0)
        self.assertIsNotNone(sc["mean_tau"])


if __name__ == "__main__":
    unittest.main()

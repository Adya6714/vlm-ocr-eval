"""Checkable properties of the HarfBuzz renderer."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renderer.degradation_profile import DegradationSample  # noqa: E402
from renderer.glyph_frequency import count_clusters, normalize_counts  # noqa: E402
from renderer.render import (  # noqa: E402
    find_font,
    render_tier_a,
    render_tier_b,
    render_tier_c,
    line_cluster_boxes,
    _hb_font,
)


TEXTS = [
    "क्षितिज पर काली घटाएँ छाई हुई हैं।",
    "राम ने सीता को किताब दी।",
    "ज्ञानी व्यक्ति शांत रहता है।",
    "भारत एक विशाल देश है।",
]


class FontTests(unittest.TestCase):
    def test_finds_devanagari_font(self):
        path, idx = find_font("deva")
        self.assertTrue(Path(path).exists())
        self.assertGreaterEqual(idx, 0)


class RenderTests(unittest.TestCase):
    def test_tier_a_under_one_second_with_gt(self):
        page = render_tier_a(TEXTS, mode="natural", rng=np.random.default_rng(0))
        self.assertLess(page.ground_truth.elapsed_ms, 1000.0)
        self.assertGreater(len(page.ground_truth.clusters), 10)
        self.assertEqual(page.ground_truth.tier, "A")
        # Image has ink
        arr = np.asarray(page.image)
        self.assertLess(arr.mean(), 254.5)
        # Every bbox is on-page
        for c in page.ground_truth.clusters:
            x0, y0, x1, y1 = c.bbox
            self.assertLessEqual(x0, x1)
            self.assertLessEqual(y0, y1)
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, page.ground_truth.width)
            self.assertLessEqual(y1, page.ground_truth.height)
        # Line GT is what Stage 2a crops from — must be non-empty and
        # page-ordered, with boxes inside the page.
        self.assertGreater(len(page.ground_truth.lines), 0)
        orders = [ln.reading_order for ln in page.ground_truth.lines]
        self.assertEqual(orders, list(range(len(orders))))
        for ln in page.ground_truth.lines:
            x0, y0, x1, y1 = ln.bbox
            self.assertLessEqual(x0, x1)
            self.assertLessEqual(y0, y1)
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, page.ground_truth.width)
            self.assertLessEqual(y1, page.ground_truth.height)
            self.assertTrue(ln.text)

    def test_conjunct_gets_a_box(self):
        page = render_tier_a(["क्ष"], mode="natural", rng=np.random.default_rng(1))
        texts = [c.text for c in page.ground_truth.clusters]
        self.assertIn("क्ष", texts)

    def test_save_roundtrip(self):
        page = render_tier_a(TEXTS[:2], mode="natural", rng=np.random.default_rng(2))
        with tempfile.TemporaryDirectory() as td:
            img_p, gt_p = page.save(td, "page0")
            self.assertTrue(img_p.exists())
            payload = json.loads(gt_p.read_text(encoding="utf-8"))
            self.assertEqual(payload["tier"], "A")
            self.assertGreater(len(payload["clusters"]), 0)
            Image.open(img_p).verify()

    def test_tier_b_applies_degradation_when_forced(self):
        deg = DegradationSample(2.0, 6.0, 1.0, 0.3, "test")
        clean = render_tier_a(
            TEXTS, mode="natural", degradation=DegradationSample(0, 0, 0, 0, "c"),
            rng=np.random.default_rng(3),
        )
        dirty = render_tier_b(
            TEXTS, mode="natural", degradation=deg, rng=np.random.default_rng(3),
        )
        self.assertGreater(
            np.mean(np.abs(np.asarray(dirty.image).astype(float) - np.asarray(clean.image))),
            1.0,
        )

    def test_tier_c_passthrough(self):
        plain = ROOT / "data" / "raw" / "hindi" / "images" / "20_plain.png"
        if not plain.exists():
            self.skipTest("GlotOCR hindi images not present")
        page = render_tier_c(plain, "दुरघटनाओं के कारण", source_id="20")
        self.assertEqual(page.ground_truth.tier, "C")
        self.assertEqual(page.ground_truth.clusters, [])
        self.assertEqual(page.image.size[0], page.ground_truth.width)

    def test_histogram_mode_changes_cluster_bag(self):
        # Same seed, different modes — realized bags should differ once
        # inverted promotes rare conjuncts in TEXTS (ज्ञ).
        nat = render_tier_a(TEXTS * 4, mode="natural", rng=np.random.default_rng(4))
        inv = render_tier_a(TEXTS * 4, mode="inverted", rng=np.random.default_rng(4))
        nat_c = normalize_counts(count_clusters([c.text for c in nat.ground_truth.clusters]))
        inv_c = normalize_counts(count_clusters([c.text for c in inv.ground_truth.clusters]))
        # At least one cluster frequency moves by a noticeable amount.
        keys = set(nat_c) | set(inv_c)
        delta = max(abs(nat_c.get(k, 0) - inv_c.get(k, 0)) for k in keys)
        self.assertGreater(delta, 0.01)


class ShapeBoxTests(unittest.TestCase):
    def test_boxes_cover_advances(self):
        path, idx = find_font("deva")
        _, font = _hb_font(path, idx)
        boxes = line_cluster_boxes("काका", font, 32.0, 10.0, 40.0)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0][0], "का")
        # Second cluster starts to the right of the first.
        self.assertGreater(boxes[1][1][0], boxes[0][1][0])


if __name__ == "__main__":
    unittest.main()

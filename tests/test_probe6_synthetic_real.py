"""Checkable properties for Probe 6 leakage gate and correctness scoring."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "src" / "probes"
sys.path.insert(0, str(PROBES_DIR))

from probe6_synthetic_real_gap import (  # noqa: E402
    assert_no_train_raw_leakage,
    collect_manifest_image_paths,
    is_correct,
)


class TestLeakageCheck(unittest.TestCase):
    def test_repo_hindi_manifests_do_not_overlap_raw(self):
        """The held-out claim for Probe 6 — must stay true in this checkout."""
        manifests = ROOT / "data" / "manifests"
        raw = ROOT / "data" / "raw" / "hindi" / "images"
        if not manifests.exists() or not raw.exists():
            self.skipTest("data/manifests or data/raw/hindi/images missing")
        summary = assert_no_train_raw_leakage(manifests, raw, "hindi")
        self.assertTrue(summary["leakage_free"])
        self.assertEqual(summary["n_overlaps"], 0)
        self.assertGreater(summary["n_raw_images"], 0)

    def test_detects_injected_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            man_dir = tmp_path / "manifests"
            raw_dir = tmp_path / "raw" / "hindi" / "images"
            man_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            (raw_dir / "leak_plain.png").write_bytes(b"x")
            # Only natural manifest needed for the checker loop.
            for cond in ("natural", "flattened", "inverted"):
                rows = [{"image_path": "data/cache/ok.png", "text": "अ"}]
                if cond == "natural":
                    rows.append({
                        "image_path": "data/raw/hindi/images/leak_plain.png",
                        "text": "आ",
                    })
                (man_dir / f"hindi_{cond}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in rows) + "\n",
                    encoding="utf-8",
                )
            with self.assertRaises(RuntimeError):
                assert_no_train_raw_leakage(man_dir, raw_dir, "hindi")


class TestIsCorrect(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(is_correct("हिन्दी", "हिन्दी", "hindi"))

    def test_mismatch(self):
        self.assertFalse(is_correct("हिन्दी", "बंगाली", "hindi"))


class TestCollectManifest(unittest.TestCase):
    def test_reads_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.jsonl"
            p.write_text(
                json.dumps({"image_path": "a.png", "text": "x"}) + "\n"
                + json.dumps({"image_path": "b.png", "text": "y"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(collect_manifest_image_paths(p), {"a.png", "b.png"})


if __name__ == "__main__":
    unittest.main()

"""Checkable properties of Probe 5b charset composition and task building."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "src" / "probes"
sys.path.insert(0, str(PROBES_DIR))

from probe5b_zeroshot_floor import (  # noqa: E402
    charset_composition,
    classify_grapheme,
    load_completed_keys,
    resolve_repo_root,
)


class TestCharsetComposition(unittest.TestCase):
    def test_pure_devanagari_counts_as_trained(self):
        text = "हिन्दी भाषा"
        comp = charset_composition(text, "devanagari", "ol_chiki")
        self.assertGreater(comp["n_graphemes"], 0)
        self.assertEqual(comp["trained_script_fraction"], 1.0)
        self.assertEqual(comp["image_script_fraction"], 0.0)

    def test_ol_chiki_in_output_counts_as_image_script(self):
        text = "ᱯᱟᱭᱞᱚᱴ ᱫᱚ"
        comp = charset_composition(text, "devanagari", "ol_chiki")
        self.assertEqual(comp["image_script_fraction"], 1.0)
        self.assertEqual(comp["trained_script_fraction"], 0.0)

    def test_arabic_output_on_kashmiri_image(self):
        text = "ممی، پَکھ"
        comp = charset_composition(text, "devanagari", "arabic")
        self.assertEqual(comp["image_script_fraction"], 1.0)
        self.assertEqual(comp["trained_script_fraction"], 0.0)

    def test_mixed_output_splits_fractions(self):
        # One Devanagari cluster + one Ol Chiki cluster (approximate with known chars)
        text = "ह" + "ᱯ"
        comp = charset_composition(text, "devanagari", "ol_chiki")
        self.assertEqual(comp["n_graphemes"], 2)
        self.assertAlmostEqual(comp["trained_script_fraction"], 0.5)
        self.assertAlmostEqual(comp["image_script_fraction"], 0.5)

    def test_whitespace_excluded_from_denominator(self):
        text = "हिन्दी   भाषा"
        comp = charset_composition(text, "devanagari", "ol_chiki")
        self.assertEqual(comp["n_other"], 0)
        self.assertEqual(comp["trained_script_fraction"], 1.0)

    def test_empty_output_returns_none_fractions(self):
        comp = charset_composition("", "devanagari", "ol_chiki")
        self.assertIsNone(comp["trained_script_fraction"])
        self.assertEqual(comp["n_graphemes"], 0)

    def test_classify_grapheme_latin_is_other(self):
        self.assertEqual(classify_grapheme("A", "devanagari", "ol_chiki"), "other")


class TestProbe5bHelpers(unittest.TestCase):
    def test_resolve_repo_root_when_data_root_is_data(self):
        self.assertEqual(resolve_repo_root(Path("/repo/data")), Path("/repo"))

    def test_load_completed_keys_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            row = {
                "condition": "santhali",
                "image_path": "/images/0_plain.png",
            }
            out.write_text(json.dumps(row) + "\n", encoding="utf-8")
            keys = load_completed_keys(out)
            self.assertEqual(keys, {("santhali", "/images/0_plain.png")})


if __name__ == "__main__":
    unittest.main()

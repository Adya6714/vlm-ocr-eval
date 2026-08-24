"""Checkable properties of layout extraction — known geometry in, category out."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renderer.layout_sources import (  # noqa: E402
    LayoutTemplate,
    classify_layout,
    extract_layout_from_image,
    extract_layouts_from_pdf,
    load_layout_bank,
    save_layout_bank,
    Region,
)

try:
    import pymupdf
except ImportError:
    pymupdf = None


def _blank(w=400, h=600, color=255):
    return Image.new("L", (w, h), color)


class ClassifyLayoutTests(unittest.TestCase):
    def test_form_beats_two_column(self):
        regions = [
            Region("column_0", 0.05, 0.1, 0.4, 0.8, 0, "text"),
            Region("column_1", 0.55, 0.1, 0.4, 0.8, 1, "text"),
            Region("field_0", 0.1, 0.2, 0.3, 0.05, 2, "form_field"),
        ]
        self.assertEqual(classify_layout(regions, n_widgets=1), "form")

    def test_table_embedded_from_area(self):
        regions = [
            Region("body", 0.1, 0.1, 0.8, 0.3, 0, "text"),
            Region("table_0", 0.1, 0.45, 0.8, 0.4, 1, "table"),
        ]
        self.assertEqual(classify_layout(regions), "table-embedded")

    def test_tiny_infobox_table_is_not_table_embedded(self):
        regions = [
            Region("column_0", 0.05, 0.1, 0.5, 0.8, 0, "text"),
            Region("column_1", 0.60, 0.1, 0.3, 0.7, 1, "text"),
            Region("table_0", 0.6, 0.1, 0.1, 0.03, 2, "table"),
        ]
        self.assertEqual(classify_layout(regions, n_tables=1), "two-column")

    def test_two_column_from_bodies(self):
        regions = [
            Region("column_0", 0.05, 0.1, 0.4, 0.8, 0, "text"),
            Region("column_1", 0.55, 0.1, 0.4, 0.8, 1, "text"),
        ]
        self.assertEqual(classify_layout(regions), "two-column")

    def test_marginalia_kind(self):
        regions = [
            Region("body", 0.2, 0.1, 0.7, 0.8, 0, "text"),
            Region("note", 0.02, 0.2, 0.12, 0.5, 1, "margin"),
        ]
        self.assertEqual(classify_layout(regions), "marginalia")

    def test_single_column_default(self):
        regions = [Region("body", 0.1, 0.1, 0.8, 0.8, 0, "text")]
        self.assertEqual(classify_layout(regions), "single-column")


class ImageExtractionTests(unittest.TestCase):
    def test_two_ink_columns(self):
        im = _blank()
        draw = ImageDraw.Draw(im)
        # Two dense ink blocks, left and right, with a clear gutter.
        rng = np.random.default_rng(0)
        pix = np.array(im)
        pix[80:520, 30:160] = rng.integers(0, 40, size=(440, 130))
        pix[80:520, 240:370] = rng.integers(0, 40, size=(440, 130))
        im = Image.fromarray(pix.astype(np.uint8), mode="L")
        tmpl = extract_layout_from_image(im, source="test", source_id="two_col")
        self.assertEqual(tmpl.category, "two-column")
        text_cols = [r for r in tmpl.regions if r.kind == "text"]
        self.assertGreaterEqual(len(text_cols), 2)

    def test_single_ink_column(self):
        im = _blank()
        pix = np.array(im)
        rng = np.random.default_rng(1)
        pix[60:540, 50:350] = rng.integers(0, 50, size=(480, 300))
        im = Image.fromarray(pix.astype(np.uint8), mode="L")
        tmpl = extract_layout_from_image(im, source="test", source_id="one_col")
        self.assertEqual(tmpl.category, "single-column")
        bodies = [r for r in tmpl.regions if r.kind == "text"]
        self.assertEqual(len(bodies), 1)
        self.assertGreater(bodies[0].width, 0.5)


@unittest.skipIf(pymupdf is None, "pymupdf not installed")
class PdfExtractionTests(unittest.TestCase):
    def _write_pdf(self, build_page) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "page.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        build_page(page)
        doc.save(tmp)
        doc.close()
        return tmp

    def test_two_column_pdf(self):
        filler = "alpha bravo charlie delta echo foxtrot golf hotel " * 12

        def build(page):
            page.insert_textbox(pymupdf.Rect(40, 80, 270, 780), filler, fontsize=11)
            page.insert_textbox(pymupdf.Rect(320, 80, 555, 780), filler, fontsize=11)

        path = self._write_pdf(build)
        tmpls = extract_layouts_from_pdf(path, "test", "two_col_pdf", max_pages=1)
        self.assertEqual(len(tmpls), 1)
        self.assertEqual(tmpls[0].category, "two-column")

    def test_single_column_pdf(self):
        filler = "alpha bravo charlie delta echo foxtrot golf hotel " * 20

        def build(page):
            page.insert_textbox(pymupdf.Rect(60, 80, 535, 780), filler, fontsize=12)

        path = self._write_pdf(build)
        tmpls = extract_layouts_from_pdf(path, "test", "one_col_pdf", max_pages=1)
        self.assertEqual(tmpls[0].category, "single-column")

    def test_form_widgets_pdf(self):
        def build(page):
            page.insert_textbox(pymupdf.Rect(50, 40, 400, 80), "Name")
            widget = pymupdf.Widget()
            widget.field_name = "name"
            widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
            widget.rect = pymupdf.Rect(50, 90, 300, 120)
            page.add_widget(widget)

        path = self._write_pdf(build)
        tmpls = extract_layouts_from_pdf(path, "test", "form_pdf", max_pages=1)
        self.assertEqual(tmpls[0].category, "form")
        self.assertTrue(any(r.kind == "form_field" for r in tmpls[0].regions))


class BankRoundtripTests(unittest.TestCase):
    def test_save_load(self):
        tmpl = LayoutTemplate(
            source="test",
            source_id="x",
            category="single-column",
            page_index=0,
            page_width=100,
            page_height=200,
            regions=[Region("body", 0.1, 0.1, 0.8, 0.8, 0, "text")],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bank.json"
            save_layout_bank([tmpl], path)
            loaded = load_layout_bank(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].category, "single-column")
            self.assertEqual(loaded[0].regions[0].name, "body")


if __name__ == "__main__":
    unittest.main()

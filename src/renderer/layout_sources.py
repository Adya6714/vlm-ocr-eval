"""
Layout template bank, sourced from real documents rather than hand-drawn
frames.

Why this exists: Probe 1 needs the *glyph-frequency* variable isolated,
but every other probe (and the demo model) has to see pages that look like
documents, not like a typesetting demo. A 2026 Devanagari OCR benchmark
found systems clustering tightly on clean synthetic text and spreading
across tens of points on real scans — invented templates would put this
project on the wrong side of that gap. DECISIONS.md #9 is the short form:
pull layouts from Internet Archive scans, India.gov.in PDFs, and Wikipedia
Indic articles, then store them as a reusable bank.

Where it sits: first module in Stage 1. `render.py` samples a template
from the bank and pours HarfBuzz-shaped text into its regions.
`degradation_profile.py` is independent (it measures pixel damage, not
geometry). Downstream, Stage 3 buckets reading-order difficulty by the
same category labels this file assigns (`single-column` → `form`).

Output: JSON under `data/cache/layouts/bank.json`. Fetch once, extract
many times — Colab sessions die, the bank should not need to.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    import pymupdf
except ImportError:  # pragma: no cover - exercised only when the dep is missing
    pymupdf = None


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "layouts"
BANK_PATH = CACHE_DIR / "bank.json"
PDF_DIR = CACHE_DIR / "pdfs"
IMAGE_DIR = CACHE_DIR / "pages"

# Wikipedia REST and MediaWiki require a descriptive UA; anonymous
# Python-urllib gets 403. This project name is enough for a research UA.
USER_AGENT = "ocr-vlm-eval/0.1 (Indic OCR research renderer; layout extraction)"

# Seed sources. These are *starting* identifiers, not a claim that they
# are the only layouts the bank will ever hold. `build_layout_bank` also
# swallows anything already sitting in PDF_DIR / IMAGE_DIR so a later
# session can drop in a prescription scan without code changes.
WIKIPEDIA_PAGES = [
    ("hi", "भारत"),
    ("bn", "বাংলাদেশ"),
    ("hi", "महाभारत"),
]

# Digital Library of India / IA items known to be Hindi or Bengali scans.
# Page images are pulled via IIIF at a modest width so we extract layout
# without downloading a 200MB PDF.
INTERNET_ARCHIVE_IDS = [
    "in.ernet.dli.2015.480257",  # Sanskrit Aur Hindi (DLI, Hindi)
    "shrimad-bhagwat-geeta-hindi-sanskrit-gorkhpur-press",
]

# Public government PDFs. india.gov.in itself is a portal; the actual
# multilingual file often lives on a .gov.in host linked from it. URLs
# are fetched with a cache-first policy — a 404 here does not fail the
# bank, it just skips that source.
INDIA_GOV_PDF_URLS = [
    # NCERT Hindi textbook chapter — real school-book layout (heads,
    # body, occasional exercises/tables), linked from the national
    # education stack rather than invented.
    "https://ncert.nic.in/textbook/pdf/jhhn101.pdf",
]

CATEGORIES = (
    "form",
    "table-embedded",
    "two-column",
    "marginalia",
    "single-column",
)


@dataclass
class Region:
    """
    One place on a page where text (or a table, or a form field) lives.

    Coordinates are fractions of page width/height, not pixels, so the
    same template can be painted at 512px or 2048px in `render.py`
    without a second extraction pass. `reading_order` is the sequence
    a human would follow; Stage 3's Kendall-tau metric will score model
    output against this, not against "top-to-bottom, left-to-right."
    """

    name: str
    x: float
    y: float
    width: float
    height: float
    reading_order: int
    kind: str = "text"  # text | table | form_field | header | footer | margin


@dataclass
class LayoutTemplate:
    """
    A reusable page geometry pulled off one real document page.

    `category` is one of CATEGORIES and is the axis Stage 3 buckets on.
    `source` is a provenance tag (internet_archive / india_gov /
    wikipedia / local), not used for sampling — `category` is.
    """

    source: str
    source_id: str
    category: str
    page_index: int
    page_width: int
    page_height: int
    regions: list[Region] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "LayoutTemplate":
        regions = [Region(**r) for r in payload.get("regions", [])]
        data = dict(payload)
        data["regions"] = regions
        return cls(**data)


def _ensure_cache_dirs() -> None:
    """Colab-safe: creating the cache dirs is the resume path, not a setup step."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _http_get(url: str, dest: Path | None = None, timeout: int = 60) -> bytes:
    """
    Tiny cache-first GET.

    Why not requests.Session: one less dependency on Colab, and the
    renderer must still work if the fetch fails — callers treat an
    exception as "skip this source," never as a hard crash of the bank.
    """
    if dest is not None and dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def _normalize_box(x0: float, y0: float, x1: float, y1: float,
                   page_w: float, page_h: float) -> tuple[float, float, float, float]:
    """Pixel bbox → page-fraction bbox, clamped to [0, 1]."""
    pw = max(page_w, 1.0)
    ph = max(page_h, 1.0)
    x = max(0.0, min(1.0, x0 / pw))
    y = max(0.0, min(1.0, y0 / ph))
    w = max(0.0, min(1.0 - x, (x1 - x0) / pw))
    h = max(0.0, min(1.0 - y, (y1 - y0) / ph))
    return x, y, w, h


def _projection_bands(values: np.ndarray, min_width: int, min_mass_frac: float) -> list[tuple[int, int]]:
    """
    Split a 1-D ink projection into contiguous occupied bands.

    Used for both column detection (vertical projection) and line-block
    detection (horizontal projection). A gap is "real" only if it is at
    least `min_width` bins, otherwise we would split a single column on
    a ragged margin. `min_mass_frac` drops speckle (a stray stamp, a
    page number) so it does not become its own column.
    """
    total = float(values.sum())
    if total <= 0:
        return []
    threshold = 0.08 * float(np.percentile(values[values > 0], 50)) if np.any(values > 0) else 0.0
    occupied = values > threshold
    bands: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(occupied):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_width:
                bands.append((start, i))
            start = None
    if start is not None and len(values) - start >= min_width:
        bands.append((start, len(values)))

    massy = []
    for a, b in bands:
        mass = float(values[a:b].sum()) / total
        if mass >= min_mass_frac:
            massy.append((a, b))
    return massy


def classify_layout(regions: Sequence[Region], n_tables: int = 0, n_widgets: int = 0) -> str:
    """
    Map a set of regions onto the five-category bank Stage 3 expects.

    Priority is specific → general: a form *looks* two-column but the
    thing we care about is the field grid; a page with a real table is
    `table-embedded` even if the table sits in one column. Getting this
    order wrong would hide the hard cases inside `single-column` and
    flatten the tau-vs-complexity curve.

    Called from both the PDF extractor and the image extractor so a
    scanned IA page and a born-digital gov PDF share labels.
    """
    if n_widgets > 0 or any(r.kind == "form_field" for r in regions):
        return "form"
    table_area = sum(r.width * r.height for r in regions if r.kind == "table")
    has_real_table = any(r.kind == "table" and r.height >= 0.18 and r.width >= 0.25 for r in regions)
    # n_tables alone is not enough: Wikipedia print PDFs emit tiny
    # infobox/navbox tables that would otherwise swallow every article
    # into `table-embedded` and erase the two-column structure that is
    # the actual reading-order problem.
    if has_real_table or table_area >= 0.08:
        return "table-embedded"

    text_regions = [r for r in regions if r.kind in {"text", "header", "footer"}]
    if not text_regions:
        text_regions = list(regions)

    # Two columns: two body regions whose x-centers sit in different
    # halves and whose widths are each a column, not a full page.
    bodies = [r for r in text_regions if r.kind == "text" and r.width >= 0.20 and r.height >= 0.15]
    if len(bodies) >= 2:
        centers = sorted(r.x + r.width / 2 for r in bodies)
        if centers[-1] - centers[0] >= 0.25:
            return "two-column"

    if any(r.kind == "margin" for r in regions):
        return "marginalia"
    # Also treat a thin tall region hugging the edge as marginalia even
    # if the extractor labelled it "text" — scanned books do this.
    for r in regions:
        hugs_edge = r.x < 0.08 or (r.x + r.width) > 0.92
        thin_tall = r.width < 0.18 and r.height > 0.20
        if hugs_edge and thin_tall:
            return "marginalia"
    return "single-column"


def extract_layout_from_image(
    image: Image.Image | np.ndarray,
    source: str = "local",
    source_id: str = "image",
    page_index: int = 0,
) -> LayoutTemplate:
    """
    Infer column / margin geometry from a *scanned* page, with no text layer.

    Internet Archive Indic books are photographs of paper. There is no
    PDF text stream to ask "where are the columns?" so we fall back to
    the oldest reliable trick in document analysis: ink projection
    profiles. This is deliberately not a learned layout model — Stage 1
    needs a deterministic, inspectable bank, not another neural net
    whose errors would leak into Probe 1.

    Hands off to: `classify_layout`, then the JSON bank. `render.py`
    ignores the original pixels and only reuses the boxes.
    """
    if isinstance(image, Image.Image):
        page_w, page_h = image.size
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            gray = arr.mean(axis=2).astype(np.float32)
        else:
            gray = arr.astype(np.float32)
        page_h, page_w = gray.shape[:2]

    # Ink is dark on paper. Otsu would need cv2; a percentile cut is
    # stable on both clean Wikipedia PDFs and stained IA scans.
    ink = gray < np.percentile(gray, 35)
    vert = ink.sum(axis=0).astype(np.float64)
    horiz = ink.sum(axis=1).astype(np.float64)

    col_bands = _projection_bands(vert, min_width=max(8, page_w // 40), min_mass_frac=0.08)
    row_bands = _projection_bands(horiz, min_width=max(8, page_h // 60), min_mass_frac=0.04)

    regions: list[Region] = []
    order = 0

    # Header / footer: first and last horizontal bands if they are short.
    if row_bands:
        y0, y1 = row_bands[0]
        if (y1 - y0) / page_h < 0.12:
            x, y, w, h = _normalize_box(0, y0, page_w, y1, page_w, page_h)
            regions.append(Region("header", x, y, w, h, order, "header"))
            order += 1
            row_bands = row_bands[1:]
        y0, y1 = row_bands[-1] if row_bands else (0, 0)
        if row_bands and (y1 - y0) / page_h < 0.10:
            x, y, w, h = _normalize_box(0, y0, page_w, y1, page_w, page_h)
            regions.append(Region("footer", x, y, w, h, order, "footer"))
            row_bands = row_bands[:-1]

    body_top = row_bands[0][0] if row_bands else int(0.08 * page_h)
    body_bot = row_bands[-1][1] if row_bands else int(0.95 * page_h)

    if len(col_bands) >= 2:
        # The leftmost or rightmost band is marginalia if it is much
        # thinner than the others; otherwise they are equal columns.
        widths = [b - a for a, b in col_bands]
        median_w = float(np.median(widths))
        for i, (a, b) in enumerate(col_bands):
            x, y, w, h = _normalize_box(a, body_top, b, body_bot, page_w, page_h)
            kind = "text"
            name = f"column_{i}"
            if (b - a) < 0.45 * median_w or w < 0.16:
                kind = "margin"
                name = f"margin_{i}"
            regions.append(Region(name, x, y, w, h, order, kind))
            order += 1
    elif len(col_bands) == 1:
        a, b = col_bands[0]
        x, y, w, h = _normalize_box(a, body_top, b, body_bot, page_w, page_h)
        regions.append(Region("body", x, y, w, h, order, "text"))
        order += 1
    else:
        x, y, w, h = _normalize_box(
            int(0.08 * page_w), body_top, int(0.92 * page_w), body_bot, page_w, page_h
        )
        regions.append(Region("body", x, y, w, h, 0, "text"))

    category = classify_layout(regions)
    return LayoutTemplate(
        source=source,
        source_id=source_id,
        category=category,
        page_index=page_index,
        page_width=int(page_w),
        page_height=int(page_h),
        regions=regions,
    )


def extract_layout_from_pdf_page(page, source: str, source_id: str, page_index: int) -> LayoutTemplate:
    """
    Pull region boxes from a born-digital PDF page (Wikipedia print, gov forms).

    Why PDF here and projections on scans: a digital PDF already knows
    where its blocks are. Re-detecting them from pixels would only add
    error. Tables and AcroForm widgets are first-class in pymupdf and
    map directly onto the `table-embedded` and `form` categories.

    Called from `extract_layouts_from_pdf` once per page. Requires
    pymupdf; the image path does not.
    """
    if pymupdf is None:
        raise RuntimeError("pymupdf is required for PDF layout extraction")

    page_w, page_h = float(page.rect.width), float(page.rect.height)
    regions: list[Region] = []
    order = 0

    widgets = list(page.widgets() or [])
    for i, widget in enumerate(widgets):
        r = widget.rect
        x, y, w, h = _normalize_box(r.x0, r.y0, r.x1, r.y1, page_w, page_h)
        regions.append(Region(f"field_{i}", x, y, w, h, order, "form_field"))
        order += 1

    n_tables = 0
    try:
        found = page.find_tables()
        tables = list(found.tables) if found is not None else []
    except Exception:
        tables = []
    for i, table in enumerate(tables):
        r = table.bbox
        x, y, w, h = _normalize_box(r[0], r[1], r[2], r[3], page_w, page_h)
        regions.append(Region(f"table_{i}", x, y, w, h, order, "table"))
        order += 1
        n_tables += 1

    text = page.get_text("dict")
    blocks = []
    for block in text.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block["bbox"]
        x, y, w, h = _normalize_box(bbox[0], bbox[1], bbox[2], bbox[3], page_w, page_h)
        if w * h < 0.002:
            continue
        blocks.append((x, y, w, h, bbox))

    # Cluster remaining text blocks into columns by x-center, then emit
    # one region per cluster (plus header/footer by y-position).
    if blocks:
        # A header/footer is a *short strip* at the edge, not "any block
        # whose top happens to sit in the first 10% of the page." Body
        # text on a real PDF often starts at y≈0.08; treating that as a
        # header merges both columns into one spanning box and the page
        # silently becomes single-column.
        header_blocks = [b for b in blocks if b[1] < 0.07 and b[3] < 0.08]
        footer_blocks = [b for b in blocks if (b[1] + b[3]) > 0.93 and b[3] < 0.08]
        body_blocks = [b for b in blocks if b not in header_blocks and b not in footer_blocks]

        if header_blocks:
            x0 = min(b[0] for b in header_blocks)
            y0 = min(b[1] for b in header_blocks)
            x1 = max(b[0] + b[2] for b in header_blocks)
            y1 = max(b[1] + b[3] for b in header_blocks)
            regions.append(Region("header", x0, y0, x1 - x0, y1 - y0, order, "header"))
            order += 1

        if body_blocks:
            centers = np.array([b[0] + b[2] / 2 for b in body_blocks])
            # 1-D 2-means by whether the x-center is left or right of
            # the page midpoint, then keep both clusters only if each
            # has enough mass. Cheaper and more stable than a general
            # clustering dependency for "is this two columns?"
            left = [b for b, c in zip(body_blocks, centers) if c < 0.48]
            right = [b for b, c in zip(body_blocks, centers) if c > 0.52]
            mid = [b for b, c in zip(body_blocks, centers) if 0.48 <= c <= 0.52]
            left_area = sum(b[2] * b[3] for b in left)
            right_area = sum(b[2] * b[3] for b in right)
            two_col = left and right and left_area > 0.04 and right_area > 0.04

            def _union(name, group, kind, idx):
                nonlocal order
                x0 = min(b[0] for b in group)
                y0 = min(b[1] for b in group)
                x1 = max(b[0] + b[2] for b in group)
                y1 = max(b[1] + b[3] for b in group)
                regions.append(Region(name, x0, y0, x1 - x0, y1 - y0, idx, kind))
                order = idx + 1

            if two_col:
                _union("column_0", left, "text", order)
                _union("column_1", right, "text", order)
                if mid:
                    # Mid-page blocks on a two-column layout are often
                    # a spanning title or a figure caption, not a third
                    # column — keep them as their own region so render
                    # doesn't pour body text through the gutter.
                    _union("span", mid, "header", order)
            else:
                group = body_blocks
                x0 = min(b[0] for b in group)
                # Thin leftover on the far left/right of a mostly
                # single-column page is marginalia (side notes, folio).
                side = [b for b in group if b[0] < 0.07 or (b[0] + b[2]) > 0.93]
                main = [b for b in group if b not in side]
                if main and side and sum(b[2] * b[3] for b in side) < 0.5 * sum(b[2] * b[3] for b in main):
                    _union("body", main, "text", order)
                    _union("margin", side, "margin", order)
                else:
                    _union("body", group, "text", order)

        if footer_blocks:
            x0 = min(b[0] for b in footer_blocks)
            y0 = min(b[1] for b in footer_blocks)
            x1 = max(b[0] + b[2] for b in footer_blocks)
            y1 = max(b[1] + b[3] for b in footer_blocks)
            regions.append(Region("footer", x0, y0, x1 - x0, y1 - y0, order, "footer"))

    if not regions:
        regions.append(Region("body", 0.08, 0.08, 0.84, 0.84, 0, "text"))

    category = classify_layout(regions, n_tables=n_tables, n_widgets=len(widgets))
    return LayoutTemplate(
        source=source,
        source_id=source_id,
        category=category,
        page_index=page_index,
        page_width=int(page_w),
        page_height=int(page_h),
        regions=regions,
    )


def extract_layouts_from_pdf(pdf_path: Path | str, source: str, source_id: str,
                             max_pages: int = 8) -> list[LayoutTemplate]:
    """
    Walk a cached PDF and emit one template per page (capped).

    Cap is on purpose: a 400-page DLI book would dominate the bank with
    near-duplicate single-column pages and drown the rare table/form
    geometries we actually need variety of. Eight pages per source is
    enough to catch front-matter vs. body vs. a table page.
    """
    if pymupdf is None:
        raise RuntimeError("pymupdf is required for PDF layout extraction")
    templates = []
    doc = pymupdf.open(pdf_path)
    try:
        n = min(len(doc), max_pages)
        for i in range(n):
            templates.append(
                extract_layout_from_pdf_page(doc[i], source, source_id, i)
            )
    finally:
        doc.close()
    return templates


def wikipedia_pdf_url(lang: str, title: str) -> str:
    """REST endpoint that returns a printable PDF of one article."""
    encoded = urllib.parse.quote(title, safe="")
    return f"https://{lang}.wikipedia.org/api/rest_v1/page/pdf/{encoded}"


def fetch_wikipedia_pdf(lang: str, title: str) -> Path | None:
    """
    Download one Indic Wikipedia article as a PDF, cache it, return the path.

    Wikipedia-as-PDF is the "web-doc layout" source in DECISIONS.md #9:
    infobox on the right, lead at the top, sections below — a real
    reading-order problem, not a book page. Cached under PDF_DIR so a
    dead Colab session does not re-hit Wikimedia.
    """
    _ensure_cache_dirs()
    slug = re.sub(r"[^\w\-]+", "_", f"wiki_{lang}_{title}", flags=re.UNICODE)
    dest = PDF_DIR / f"{slug}.pdf"
    url = wikipedia_pdf_url(lang, title)
    try:
        _http_get(url, dest=dest, timeout=90)
        return dest
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def fetch_internet_archive_pages(
    identifier: str,
    n_pages: int = 4,
    width: int = 800,
    dest_dir: Path | None = None,
) -> list[Path]:
    """
    Pull a handful of page images from an IA item via the IIIF manifest.

    Why images, not the PDF: DLI PDFs are often 100MB+ of poorly
    layered scans. IIIF lets us ask for page 3 at 800px wide and get
    exactly that. Layout extraction then uses the projection-profile
    path, which is the one that matches how these pages were made
    (cameras, not typesetters).
    """
    _ensure_cache_dirs()
    dest_dir = Path(dest_dir) if dest_dir is not None else IMAGE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    manifest_url = f"https://iiif.archive.org/iiif/{identifier}/manifest.json"
    try:
        raw = _http_get(manifest_url, dest=IMAGE_DIR / f"{identifier}_manifest.json", timeout=60)
        manifest = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return out

    canvases = (
        manifest.get("sequences", [{}])[0].get("canvases", [])
        or manifest.get("items", [])
    )
    saved = 0
    for i, canvas in enumerate(canvases):
        if saved >= n_pages:
            break
        image_url = _iiif_image_url(canvas, width=width)
        if not image_url:
            continue
        dest = dest_dir / f"{identifier}_p{i}_w{width}.jpg"
        try:
            _http_get(image_url, dest=dest, timeout=60)
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        if dest.exists() and dest.stat().st_size > 0:
            out.append(dest)
            saved += 1
    return out


def _iiif_image_url(canvas: dict, width: int) -> str | None:
    """
    IIIF Presentation 2 and 3 put the image URL in different places.
    Archive.org currently serves v2 manifests but has flipped before;
    handle both so a silent schema change does not empty the bank.
    """
    # v2
    try:
        service = canvas["images"][0]["resource"]["service"]
        base = service.get("@id") or service.get("id")
        if base:
            return f"{base}/full/{width},/0/default.jpg"
    except (KeyError, IndexError, TypeError):
        pass
    # v3
    try:
        body = canvas["items"][0]["items"][0]["body"]
        service = body["service"]
        if isinstance(service, list):
            service = service[0]
        base = service.get("id") or service.get("@id")
        if base:
            return f"{base}/full/{width},/0/default.jpg"
    except (KeyError, IndexError, TypeError):
        pass
    return None


def fetch_india_gov_pdf(url: str) -> Path | None:
    """
    Cache a government PDF. Failure is non-fatal: gov hosts are flaky
    from Colab IPs, and the bank is still valid with Wikipedia + IA.
    """
    _ensure_cache_dirs()
    name = re.sub(r"[^\w.\-]+", "_", url.rstrip("/").split("/")[-1]) or "gov.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    dest = PDF_DIR / name
    try:
        _http_get(url, dest=dest, timeout=90)
        return dest
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def load_layout_bank(path: Path | str | None = None) -> list[LayoutTemplate]:
    """Read the on-disk bank. Empty list if nobody has built it yet — callers sample defensively."""
    path = Path(path) if path is not None else BANK_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [LayoutTemplate.from_dict(item) for item in payload]


def save_layout_bank(templates: Iterable[LayoutTemplate], path: Path | str | None = None) -> Path:
    """Atomic-enough write of the bank (write tmp then replace) so a killed session does not corrupt JSON."""
    path = Path(path) if path is not None else BANK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [t.to_dict() for t in templates]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def templates_by_category(templates: Sequence[LayoutTemplate]) -> dict[str, list[LayoutTemplate]]:
    """
    Index the bank the way `render.py` will sample it: one list per
    complexity bucket, so Probe 1 can hold layout category fixed while
    glyph frequency varies, and Stage 3 can hold frequency fixed while
    category varies.
    """
    out = {c: [] for c in CATEGORIES}
    for t in templates:
        out.setdefault(t.category, []).append(t)
    return out


def build_layout_bank(
    *,
    fetch: bool = True,
    extra_pdfs: Sequence[Path | str] | None = None,
    extra_images: Sequence[Path | str] | None = None,
    max_pages_per_pdf: int = 8,
) -> list[LayoutTemplate]:
    """
    End-to-end: optionally fetch the three real sources, extract every
    cached PDF/image, write `bank.json`, return the templates.

    `fetch=False` is the Colab-resume path — if PDFs are already in
    `data/cache/layouts/pdfs/`, we just re-extract. Extra local files
    let a later session drop in the ~20 real scans (prescriptions,
    photocopies) without editing this module.
    """
    _ensure_cache_dirs()
    templates: list[LayoutTemplate] = []
    extracted_pdfs: set[Path] = set()
    extracted_images: set[Path] = set()

    if fetch:
        for lang, title in WIKIPEDIA_PAGES:
            pdf = fetch_wikipedia_pdf(lang, title)
            if pdf is not None and pymupdf is not None:
                templates.extend(
                    extract_layouts_from_pdf(pdf, "wikipedia", f"{lang}:{title}", max_pages_per_pdf)
                )
                extracted_pdfs.add(pdf.resolve())
        for ia_id in INTERNET_ARCHIVE_IDS:
            for img_path in fetch_internet_archive_pages(ia_id):
                with Image.open(img_path) as im:
                    templates.append(
                        extract_layout_from_image(im, "internet_archive", f"{ia_id}:{img_path.name}")
                    )
                extracted_images.add(img_path.resolve())
        for url in INDIA_GOV_PDF_URLS:
            pdf = fetch_india_gov_pdf(url)
            if pdf is not None and pymupdf is not None:
                templates.extend(
                    extract_layouts_from_pdf(pdf, "india_gov", url, max_pages_per_pdf)
                )
                extracted_pdfs.add(pdf.resolve())

    # Re-extract anything already cached or passed in that this run has
    # not already consumed — the Colab-resume path (`fetch=False`) lives
    # here, as does "I dropped a prescription scan in IMAGE_DIR."
    seen_pdfs = {p.resolve() for p in PDF_DIR.glob("*.pdf")} if PDF_DIR.exists() else set()
    if extra_pdfs:
        for p in extra_pdfs:
            seen_pdfs.add(Path(p).resolve())
    if pymupdf is not None:
        for pdf in sorted(seen_pdfs):
            if pdf in extracted_pdfs:
                continue
            source = "local"
            if "wiki_" in pdf.name:
                source = "wikipedia"
            elif pdf.name in {Path(u).name for u in INDIA_GOV_PDF_URLS}:
                source = "india_gov"
            templates.extend(
                extract_layouts_from_pdf(pdf, source, pdf.name, max_pages_per_pdf)
            )

    image_paths = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
    if extra_images:
        image_paths.extend(Path(p) for p in extra_images)
    for img_path in image_paths:
        resolved = Path(img_path).resolve()
        if resolved in extracted_images:
            continue
        with Image.open(resolved) as im:
            templates.append(
                extract_layout_from_image(im, "local", resolved.name)
            )

    save_layout_bank(templates)
    return templates


def main() -> None:
    """CLI: build or rebuild the bank. Safe to re-run; HTTP is cache-first."""
    templates = build_layout_bank(fetch=True)
    by_cat = templates_by_category(templates)
    print(f"wrote {BANK_PATH} ({len(templates)} templates)")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(by_cat.get(cat, []))}")


if __name__ == "__main__":
    main()

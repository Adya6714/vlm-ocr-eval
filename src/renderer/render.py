"""
HarfBuzz-backed page renderer: text in, image + grapheme-cluster GT out.

Why this exists: everything Probe 1 needs — controlled glyph exposure,
exact per-cluster boxes, known layout complexity — has to come from a
renderer we own. Calling someone else's OCR corpus gives us images but
not the dial. This module is the Stage 1 acceptance surface: given a
script, layout, and glyph-frequency mode, produce a page in under a
second whose realized cluster histogram matches the mode.

Where it sits: last Stage 1 module. Pulls layouts from
`layout_sources.py`, degradation from `degradation_profile.py`, and
resampled text from `glyph_frequency.py`. Hands `(image, ground_truth)`
to Stage 2 training and to every probe. Do not hand-roll Indic conjunct
placement — HarfBuzz is the shaping engine (BOOK.md Chapter 2); Pillow
paints the shaped glyphs.

Tiers (IMPLEMENTATION.md Stage 1):
  A — fixed font, fixed degradation, only glyph frequency varies
  B — fonts / layouts / degradation sampled from the measured pools
  C — real documents with existing ground truth (passthrough)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import uharfbuzz as hb
from PIL import Image, ImageDraw, ImageFont

from renderer.degradation_profile import (
    DegradationProfile,
    DegradationSample,
    apply_degradation,
    load_profile,
)
from renderer.glyph_frequency import (
    Mode,
    grapheme_clusters,
    resample_corpus,
)
from renderer.layout_sources import (
    LayoutTemplate,
    Region,
    load_layout_bank,
    templates_by_category,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "cache" / "renders"

Tier = Literal["A", "B", "C"]

# Ordered candidates: first hit wins. macOS ships Kohinoor / Sangam;
# Colab / Linux typically have Noto. TTC files need an index.
FONT_CANDIDATES: dict[str, list[tuple[str, int]]] = {
    "deva": [
        ("/System/Library/Fonts/Kohinoor.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc", 0),
        ("/System/Library/Fonts/Supplemental/ITFDevanagari.ttc", 0),
        ("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf", 0),
    ],
    "beng": [
        ("/System/Library/Fonts/KohinoorBangla.ttc", 0),
        ("/System/Library/Fonts/Supplemental/Bangla Sangam MN.ttc", 0),
        ("/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf", 0),
    ],
}

SCRIPT_ALIASES = {
    "deva": "deva",
    "devanagari": "deva",
    "hindi": "deva",
    "hi": "deva",
    "beng": "beng",
    "bengali": "beng",
    "bn": "beng",
}


@dataclass
class ClusterGT:
    """One grapheme cluster on the page, with its pixel bounding box."""

    text: str
    bbox: list[int]  # [x0, y0, x1, y1]
    region: str
    line: int
    reading_order: int

@dataclass
class LineGT:
    """One wrapped line on the page — the unit Stage 2a's manifest needs."""
    text: str
    bbox: list[int]  # [x0, y0, x1, y1], padded
    region: str
    line_index: int
    reading_order: int

@dataclass
class PageGT:
    """Exact ground truth for one rendered page — what Stage 2 trains against."""

    text: str
    clusters: list[ClusterGT]
    width: int
    height: int
    tier: str
    mode: str
    layout_category: str
    layout_source: str
    font_path: str
    degradation: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0
    lines: list[LineGT] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


@dataclass
class RenderedPage:
    image: Image.Image
    ground_truth: PageGT

    def save(self, out_dir: Path | str, stem: str) -> tuple[Path, Path]:
        """Write `{stem}.png` + `{stem}.json`. Checkpoint pair for Colab resume."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{stem}.png"
        gt_path = out_dir / f"{stem}.json"
        self.image.save(img_path)
        gt_path.write_text(
            json.dumps(self.ground_truth.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return img_path, gt_path


def resolve_script(script: str) -> str:
    key = script.lower().strip()
    if key not in SCRIPT_ALIASES:
        raise ValueError(f"unsupported script {script!r}; known: {sorted(SCRIPT_ALIASES)}")
    return SCRIPT_ALIASES[key]


def find_font(script: str, rng: np.random.Generator | None = None) -> tuple[str, int]:
    """
    Pick a font that covers the script.

    Multiple candidates so the same code runs on a Mac laptop and a
    Colab VM without a font-install step in the critical path. Tier B
    may sample among the fonts that actually exist; Tier A always takes
    the first available so the controlled condition stays controlled.
    """
    script = resolve_script(script)
    available = [(p, i) for p, i in FONT_CANDIDATES[script] if Path(p).exists()]
    if not available:
        raise FileNotFoundError(
            f"no font found for script={script}; install Noto or Kohinoor "
            f"and/or extend FONT_CANDIDATES"
        )
    if rng is None:
        return available[0]
    return available[int(rng.integers(0, len(available)))]


def default_layout(page_width: int = 1024, page_height: int = 1448) -> LayoutTemplate:
    """
    Fallback single-column body when the layout bank has not been built
    yet. Marked `builtin` so it never pretends to be a real document —
    DECISIONS.md #9 still holds for the bank itself.
    """
    return LayoutTemplate(
        source="builtin",
        source_id="default-single-column",
        category="single-column",
        page_index=0,
        page_width=page_width,
        page_height=page_height,
        regions=[Region("body", 0.08, 0.08, 0.84, 0.84, 0, "text")],
    )


def _hb_font(font_path: str, face_index: int = 0) -> tuple[hb.Face, hb.Font]:
    data = Path(font_path).read_bytes()
    face = hb.Face(data, face_index)
    return face, hb.Font(face)


def shape_line(text: str, font: hb.Font) -> tuple[list, list]:
    """
    Run HarfBuzz on one line. Returns (glyph_infos, glyph_positions).

    This is the whole reason we do not call `draw.text` alone for
    metrics: Pillow+raqm shapes correctly for painting, but the glyph
    advances and cluster IDs we need for per-akshara boxes live here.
    """
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    return list(buf.glyph_infos), list(buf.glyph_positions)


def measure_line_width(text: str, font: hb.Font, font_size: float) -> float:
    """Advance width of a line in pixels — used by the wrapper."""
    if not text:
        return 0.0
    face = font.face
    _, positions = shape_line(text, font)
    scale = font_size / face.upem
    return sum(p.x_advance for p in positions) * scale


def wrap_text(text: str, font: hb.Font, font_size: float, max_width: float) -> list[str]:
    """
    Greedy wrap on whitespace, measuring with HarfBuzz.

    Wrapping on code points would split a conjunct across lines; wrapping
    on whitespace keeps grapheme clusters intact. If a single token is
    wider than the column (a long URL, a dense conjunct string), it is
    emitted on its own line rather than truncated — GT must stay exact.
    """
    if max_width <= 0:
        return [text]
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if measure_line_width(trial, font, font_size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def line_cluster_boxes(
    text: str,
    font: hb.Font,
    font_size: float,
    origin_x: float,
    origin_y: float,
) -> list[tuple[str, list[int]]]:
    """
    Map each grapheme cluster in `text` to a pixel bbox on the page.

    HarfBuzz `cluster` values from `Buffer.add_str` are Unicode character
    offsets into `text`. We walk grapheme clusters by character length,
    union the glyph advances whose cluster index falls inside each
    grapheme, and emit `[x0,y0,x1,y1]`. Glyph y-offsets matter for
    matras that hang above/below the baseline — the box must cover them
    or Stage 0-style grapheme CER will look at the wrong crop.
    """
    if not text:
        return []
    face = font.face
    scale = font_size / face.upem
    infos, positions = shape_line(text, font)

    # Grapheme spans as [start, end) character indices.
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for g in grapheme_clusters(text):
        spans.append((cursor, cursor + len(g), g))
        cursor += len(g)

    # Accumulate pixel extents per span index.
    boxes = [
        [np.inf, origin_y - font_size * 0.85, -np.inf, origin_y + font_size * 0.35]
        for _ in spans
    ]
    x = origin_x
    for info, pos in zip(infos, positions):
        g_idx = None
        for i, (start, end, _) in enumerate(spans):
            if start <= info.cluster < end:
                g_idx = i
                break
        gx0 = x + pos.x_offset * scale
        gy0 = origin_y - (pos.y_offset + pos.y_advance) * scale - font_size * 0.2
        gx1 = gx0 + max(pos.x_advance * scale, font_size * 0.15)
        gy1 = origin_y + font_size * 0.35
        if g_idx is not None:
            boxes[g_idx][0] = min(boxes[g_idx][0], gx0)
            boxes[g_idx][1] = min(boxes[g_idx][1], gy0)
            boxes[g_idx][2] = max(boxes[g_idx][2], gx1)
            boxes[g_idx][3] = max(boxes[g_idx][3], gy1)
        x += pos.x_advance * scale
        # y_advance is almost always 0 for horizontal Indic runs

    out = []
    for (start, end, g), box in zip(spans, boxes):
        if g.strip() == "":
            # Keep spaces out of the cluster GT list — they are not
            # vocabulary tokens for the instrument — but they still
            # advanced the pen above.
            continue
        if not np.isfinite(box[0]):
            continue
        out.append((g, [int(round(v)) for v in box]))
    return out


def _region_pixels(region: Region, page_w: int, page_h: int) -> tuple[int, int, int, int]:
    x0 = int(region.x * page_w)
    y0 = int(region.y * page_h)
    x1 = int((region.x + region.width) * page_w)
    y1 = int((region.y + region.height) * page_h)
    return x0, y0, x1, y1


def render_page(
    texts: Sequence[str],
    *,
    script: str = "deva",
    mode: Mode = "natural",
    tier: Tier = "A",
    layout: LayoutTemplate | None = None,
    font_path: str | None = None,
    font_index: int = 0,
    page_size: tuple[int, int] = (1024, 1448),
    font_size: float = 28.0,
    degradation: DegradationSample | None = None,
    profile: DegradationProfile | None = None,
    rng: np.random.Generator | None = None,
) -> RenderedPage:
    """
    End-to-end: resample text → pour into layout → HarfBuzz-shape →
    degrade → return image + cluster GT.

    Tier A freezes font and degradation so Probe 1's only moving part
    is `mode`. Tier B samples both. Tier C is handled by
    `render_tier_c` (real documents have no synthetic paint step).
    """
    if tier == "C":
        raise ValueError("tier C is real-document passthrough; call render_tier_c()")

    t0 = time.perf_counter()
    rng = rng or np.random.default_rng()
    script_key = resolve_script(script)

    # --- glyph-frequency dial ---
    resampled = resample_corpus(list(texts), mode, rng=rng)
    page_text = "\n".join(resampled.texts)

    # --- layout ---
    if layout is None:
        bank = load_layout_bank()
        by_cat = templates_by_category(bank)
        if tier == "B" and bank:
            layout = bank[int(rng.integers(0, len(bank)))]
        elif by_cat.get("single-column"):
            layout = by_cat["single-column"][0]
        elif bank:
            layout = bank[0]
        else:
            layout = default_layout(page_size[0], page_size[1])

    page_w, page_h = page_size

    # --- font ---
    if font_path is None:
        if tier == "A":
            font_path, font_index = find_font(script_key, rng=None)
        else:
            font_path, font_index = find_font(script_key, rng=rng)
    _, hb_font = _hb_font(font_path, font_index)
    pil_font = ImageFont.truetype(font_path, size=int(font_size), index=font_index)

    # --- paint ---
    image = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    clusters: list[ClusterGT] = []
    order = 0
    text_regions = sorted(
        [r for r in layout.regions if r.kind in {"text", "header", "footer", "margin"}],
        key=lambda r: r.reading_order,
    )
    if not text_regions:
        text_regions = [Region("body", 0.08, 0.08, 0.84, 0.84, 0, "text")]

    # Distribute sentences across regions in reading order.
    sentences = list(resampled.texts)
    if not sentences:
        sentences = [""]
    per_region = max(1, len(sentences) // len(text_regions))

    # Line-level GT sits alongside cluster GT: Stage 2a's train.py
    # consumes line crops (`{"image_path","text"}`), so every painted
    # wrap needs a page-space box without a second shaping pass —
    # measure_line_width reuses the same HarfBuzz font already loaded.
    line_order = 0
    lines_gt: list[LineGT] = []

    cursor = 0
    for ri, region in enumerate(text_regions):
        x0, y0, x1, y1 = _region_pixels(region, page_w, page_h)
        max_w = max(10.0, x1 - x0 - 4)
        line_height = font_size * 1.45
        y = y0 + font_size
        if ri == len(text_regions) - 1:
            chunk = sentences[cursor:]
        else:
            chunk = sentences[cursor: cursor + per_region]
            cursor += per_region
        region_text = " ".join(chunk)
        lines = wrap_text(region_text, hb_font, font_size, max_w)
        for li, line in enumerate(lines):
            if y + font_size * 0.4 > y1:
                break
            draw.text((x0, y - font_size), line, font=pil_font, fill=(0, 0, 0))
            line_w = measure_line_width(line, hb_font, font_size)
            lines_gt.append(LineGT(
                text=line,
                bbox=[max(0, int(x0)), max(0, int(y - font_size * 1.05)),
                      min(page_w, int(x0 + line_w)), min(page_h, int(y + font_size * 0.35))],
                region=region.name,
                line_index=li,
                reading_order=line_order,
            ))
            line_order += 1
            for g, bbox in line_cluster_boxes(line, hb_font, font_size, x0, y):
                # clip to page
                bbox = [
                    max(0, min(page_w - 1, bbox[0])),
                    max(0, min(page_h - 1, bbox[1])),
                    max(0, min(page_w, bbox[2])),
                    max(0, min(page_h, bbox[3])),
                ]
                clusters.append(
                    ClusterGT(
                        text=g,
                        bbox=bbox,
                        region=region.name,
                        line=li,
                        reading_order=order,
                    )
                )
                order += 1
            y += line_height

    # --- degradation ---
    if degradation is None:
        if profile is None:
            profile = load_profile()
        if tier == "A":
            # Probe 1 holds damage constant; zero is the constant that
            # adds no vision-side confound. Pass an explicit
            # DegradationSample to freeze a non-zero profile instead.
            degradation = DegradationSample(0.0, 0.0, 0.0, 0.0, "tierA-clean")
        else:
            degradation = (
                profile.sample(rng)
                if profile.samples
                else DegradationSample(1.0, 4.0, 0.5, 0.1, "tierB-fallback")
            )

    if any(
        getattr(degradation, a) > 0
        for a in ("blur_sigma", "noise_std", "skew_degrees", "show_through")
    ):
        image = apply_degradation(image, degradation, rng=rng)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    gt = PageGT(
        text=page_text,
        clusters=clusters,
        width=page_w,
        height=page_h,
        tier=tier,
        mode=mode,
        layout_category=layout.category,
        layout_source=f"{layout.source}:{layout.source_id}",
        font_path=font_path,
        degradation=degradation.to_dict() if degradation else {},
        elapsed_ms=elapsed_ms,
        lines=lines_gt,
    )
    return RenderedPage(image=image, ground_truth=gt)


def render_tier_a(
    texts: Sequence[str],
    mode: Mode = "natural",
    script: str = "deva",
    **kwargs,
) -> RenderedPage:
    """Probe 1's controlled condition: only `mode` should change across runs."""
    return render_page(texts, script=script, mode=mode, tier="A", **kwargs)


def render_tier_b(
    texts: Sequence[str],
    mode: Mode = "natural",
    script: str = "deva",
    **kwargs,
) -> RenderedPage:
    """Probes 2–5 headroom: sample layout, font, and measured degradation."""
    return render_page(texts, script=script, mode=mode, tier="B", **kwargs)


def render_tier_c(
    image_path: Path | str,
    ground_truth_text: str,
    *,
    source_id: str = "",
) -> RenderedPage:
    """
    Reality check: an unmodified real document.

    No HarfBuzz, no synthetic degradation. Cluster boxes are unknown
    for real scans (we do not re-OCR them here), so `clusters` is empty
    and `text` carries the existing transcription. Probe 6 compares
    system accuracy on these against Tier A/B — it does not need boxes.
    """
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    gt = PageGT(
        text=ground_truth_text,
        clusters=[],
        width=image.width,
        height=image.height,
        tier="C",
        mode="natural",
        layout_category="real",
        layout_source=source_id or path.name,
        font_path="",
        degradation={},
        elapsed_ms=0.0,
    )
    return RenderedPage(image=image, ground_truth=gt)


def main() -> None:
    """Smoke: render one Tier A page from Hindi GlotOCR lines if present."""
    gt_path = REPO_ROOT / "data" / "raw" / "hindi" / "ground_truth.jsonl"
    texts = []
    if gt_path.exists():
        for line in gt_path.read_text(encoding="utf-8").splitlines()[:12]:
            texts.append(json.loads(line)["text"])
    else:
        texts = ["क्षितिज पर काली घटाएँ छाई हुई हैं।", "राम ने सीता को किताब दी।"]

    page = render_tier_a(texts, mode="natural", script="deva")
    img_path, json_path = page.save(OUTPUT_DIR, "smoke_tierA_natural")
    print(f"wrote {img_path} and {json_path}")
    print(f"clusters={len(page.ground_truth.clusters)} elapsed_ms={page.ground_truth.elapsed_ms:.1f}")
    print(f"tv_mode_check: mode={page.ground_truth.mode} tier={page.ground_truth.tier}")


if __name__ == "__main__":
    main()

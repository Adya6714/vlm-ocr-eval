"""
Empirical scan-degradation distribution, measured off real pages.

Why this exists: Probe 1 (Tier A) has to hold degradation fixed so the
only moving part is glyph frequency. Probes 2–5 (Tier B) need the
opposite — pages that are damaged the way Indian paper actually is
damaged, not with a guessed GaussianBlur(radius=1.2). Hardcoding those
numbers would make the "degraded" condition a fiction; measuring them
off real scans makes it a distribution we can sample and, later, defend.

Where it sits: second Stage 1 module. Independent of `layout_sources.py`
(geometry vs. pixel damage). `render.py` paints a clean HarfBuzz page,
then `apply_degradation` here. Stage 5's transfer analysis runs on both
the clean draw and this sampled damage (DECISIONS.md #15).

The four parameters match what you can see on a photocopied form:
blur (the copier was a bit out of focus), additive noise (sensor /
toner grit), skew (the page went in crooked), show-through (the reverse
side bleeding through thin paper). Each is stored as the *apply*
parameter `render.py` will use, inverted from no-reference measurements
against a calibration curve built from sharp pages.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from renderer.layout_sources import INTERNET_ARCHIVE_IDS, fetch_internet_archive_pages
except ImportError:  # running as a script from src/renderer/
    from layout_sources import INTERNET_ARCHIVE_IDS, fetch_internet_archive_pages


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "degradation"
PROFILE_PATH = CACHE_DIR / "profile.json"
SCAN_DIR = CACHE_DIR / "scans"

# GlotOCR's old-document renders are a bootstrap if we have fewer than
# 20 true scans. They are not prescriptions; see DECISIONS.md #22.
GLOTOCR_IMAGE_DIRS = [
    REPO_ROOT / "data" / "raw" / "hindi" / "images",
    REPO_ROOT / "data" / "raw" / "bengali" / "images",
]


@dataclass
class DegradationSample:
    """
    One draw from the measured distribution, in renderer-ready units.

    blur_sigma: PIL GaussianBlur radius (pixels at the measured scale).
    noise_std: additive Gaussian std on 0–255.
    skew_degrees: CCW rotation to apply to a straight page.
    show_through: alpha in [0, 1] of a faint, shifted, inverted copy.
    """

    blur_sigma: float
    noise_std: float
    skew_degrees: float
    show_through: float
    source_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DegradationSample":
        return cls(**payload)


def _as_gray(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.float32)
    arr = np.asarray(image)
    if arr.ndim == 3:
        return arr.mean(axis=2).astype(np.float32)
    return arr.astype(np.float32)


def laplacian_variance(gray: np.ndarray) -> float:
    """
    Sharpness proxy. Low on blurry scans, high on hard-edged digital
    renders. Used as the observable we invert to get `blur_sigma`.
    OpenCV's Laplacian when available; a 3×3 numpy kernel otherwise, so
    this file still runs on a bare Colab if cv2 failed to import.
    """
    g = gray.astype(np.float64)
    if cv2 is not None:
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # pad by replication so the metric isn't dominated by the border
    padded = np.pad(g, 1, mode="edge")
    acc = np.zeros_like(g)
    for i in range(3):
        for j in range(3):
            acc += kernel[i, j] * padded[i:i + g.shape[0], j:j + g.shape[1]]
    return float(acc.var())


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Dark pixels. Percentile cut so stained paper still separates from toner."""
    return gray < np.percentile(gray, 40)


def _dilated_ink_mask(gray: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Ink plus a small halo. Noise and show-through are defined on paper;
    without the halo, character edges leak into the paper residual and
    every sharp page looks "noisy."
    """
    ink = _ink_mask(gray)
    if cv2 is not None:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return cv2.dilate(ink.astype(np.uint8), kernel, iterations=1).astype(bool)
    padded = np.pad(ink, radius, mode="constant", constant_values=False)
    out = np.zeros_like(ink, dtype=bool)
    h, w = ink.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def estimate_skew_degrees(gray: np.ndarray, search: float = 6.0, step: float = 0.5) -> float:
    """
    Deskew search: rotate the page across a small angle range and pick
    the rotation that makes the horizontal ink-projection peakiest
    (highest variance). A well-aligned text page has strong horizontal
    runs; a crooked one smears them.

    The returned value is the page's tilt (what `apply_degradation`
    should rotate a straight render by), not the deskew correction.

    Coarse 0.5° steps are enough — scanner skew is a degree-scale
    effect, and finer search would dominate runtime on 20 pages for no
    downstream gain (the instrument trains at ~512–1024px).
    """
    ink = _ink_mask(gray)
    if ink.sum() < 200:
        return 0.0
    # Crop to ink bbox so empty margin does not dilute the projection.
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    crop = gray[r0:r1, c0:c1]
    if crop.size == 0:
        return 0.0

    best_angle = 0.0
    best_score = -1.0
    angles = np.arange(-search, search + step / 2, step)
    pil = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), mode="L")
    for angle in angles:
        # Rotate by -angle: we are trying to *undo* a putative tilt of
        # `angle` degrees. The angle that undoes best is the tilt.
        rotated = pil.rotate(-float(angle), resample=Image.BILINEAR, fillcolor=255)
        arr = np.asarray(rotated, dtype=np.float32)
        proj = (arr < np.percentile(arr, 40)).sum(axis=1).astype(np.float64)
        score = float(proj.var())
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def estimate_noise_std(gray: np.ndarray) -> float:
    """
    Residual std after a light denoise, measured on *paper* not ink.

    Character edges are high-frequency and would look like enormous
    "noise" if we included them. Show-through is handled separately
    (structured, mid-frequency); this number is the unstructured grit
    we will re-apply as additive Gaussian noise.
    """
    if cv2 is not None:
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.0)
    else:
        pil = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
        denoised = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    residual = gray.astype(np.float64) - np.asarray(denoised, dtype=np.float64)
    paper = ~_dilated_ink_mask(gray)
    if paper.sum() < 100:
        return float(residual.std())
    return float(residual[paper].std())


def estimate_show_through(gray: np.ndarray) -> float:
    """
    Structured mid-frequency energy in the paper (non-ink) region.

    Show-through is not speckle: it is the ghost of the reverse side,
    a low-contrast image sitting in the background. After removing
    large-scale illumination (a wide Gaussian), whatever spatially
    coherent leftover remains is the ghost. Mapped to [0, 1] so
    `apply_degradation` can use it as a blend alpha.

    The 25.0 divisor is a unit choice, not a fitted constant: 25 gray
    levels of mid-frequency background std is "heavy show-through" on
    the scans we measured. Logged in DECISIONS.md #22 so it can be
    revisited once more prescription-style pages are in the pool.
    """
    paper = ~_dilated_ink_mask(gray, radius=3)
    bg = gray.copy()
    paper_med = float(np.median(gray[paper])) if paper.any() else 240.0
    bg[~paper] = paper_med
    if cv2 is not None:
        illum = cv2.GaussianBlur(bg, (0, 0), 25)
        mid = cv2.GaussianBlur(bg - illum, (0, 0), 2)
    else:
        pil = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), mode="L")
        illum = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=25)), dtype=np.float32)
        leftover = bg - illum
        mid = np.asarray(
            Image.fromarray(np.clip(leftover + 128, 0, 255).astype(np.uint8), mode="L")
            .filter(ImageFilter.GaussianBlur(radius=2)),
            dtype=np.float32,
        ) - 128.0
    energy = float(np.asarray(mid)[paper].std()) if paper.any() else 0.0
    return float(np.clip(energy / 25.0, 0.0, 1.0))


def build_blur_calibration(sharp_gray: np.ndarray, sigmas: Sequence[float] | None = None) -> list[tuple[float, float]]:
    """
    Map known Gaussian-blur radii to Laplacian variance on one sharp
    page. `estimate_blur_sigma` inverts this curve. Built per call from
    a real sharp image (GlotOCR `*_plain.png`) so the curve reflects
    Indic glyph edges, not a synthetic checkerboard.
    """
    if sigmas is None:
        sigmas = [0.0, 0.4, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0]
    pil = Image.fromarray(np.clip(sharp_gray, 0, 255).astype(np.uint8), mode="L")
    curve = []
    for sigma in sigmas:
        if sigma <= 0:
            blurred = sharp_gray
        else:
            blurred = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32)
        curve.append((float(sigma), laplacian_variance(blurred)))
    return curve


def estimate_blur_sigma(gray: np.ndarray, curve: Sequence[tuple[float, float]]) -> float:
    """
    Invert the calibration curve: find the sigma whose Laplacian
    variance on the sharp page best matches this scan.

    Why invert a curve instead of reporting raw Laplacian variance:
    `render.py` has to *apply* a blur, and PIL's radius is the unit
    that will actually run. A variance number is not that unit.

    If the scan is sharper than the calibration page (L higher than
    curve[0]), sigma is 0 — we do not invent sharpening.
    """
    L = laplacian_variance(gray)
    # Curve is monotonically decreasing in L as sigma grows.
    sigmas = [s for s, _ in curve]
    laps = [lv for _, lv in curve]
    if L >= laps[0]:
        return 0.0
    if L <= laps[-1]:
        return float(sigmas[-1])
    for i in range(len(laps) - 1):
        if laps[i] >= L >= laps[i + 1]:
            lo, hi = laps[i], laps[i + 1]
            t = 0.0 if hi == lo else (laps[i] - L) / (laps[i] - laps[i + 1])
            return float(sigmas[i] + t * (sigmas[i + 1] - sigmas[i]))
    return 0.0


def measure_image(
    image: Image.Image | np.ndarray,
    blur_curve: Sequence[tuple[float, float]],
    source_id: str = "",
) -> DegradationSample:
    """
    Full no-reference measurement of one page, converted into apply
    parameters. Called once per scan while fitting the profile; not
    on the hot path of rendering.
    """
    gray = _as_gray(image)
    return DegradationSample(
        blur_sigma=estimate_blur_sigma(gray, blur_curve),
        noise_std=estimate_noise_std(gray),
        skew_degrees=estimate_skew_degrees(gray),
        show_through=estimate_show_through(gray),
        source_id=source_id,
    )


def apply_degradation(
    image: Image.Image,
    sample: DegradationSample,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """
    Paint the four measured effects onto a clean renderer output.

    Order is the physical order: the page sat crooked in the scanner
    (skew), the optics were soft (blur), the sensor added grit (noise),
    and thin paper let the reverse side through (show-through).
    Reversing this order (noise then blur) would bake the grit into
    the blur and no longer match how the measurements were defined.

    Show-through is a geometrically flipped, shifted, low-contrast copy
    of *this* page, not a second document. We do not have verso images
    for the synthetic renderer; a flipped copy is the standard proxy
    and is enough to produce the ghosted-akshara look.
    """
    rng = rng or np.random.default_rng()
    out = image.convert("RGB")
    if abs(sample.skew_degrees) > 0.05:
        out = out.rotate(sample.skew_degrees, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))
    if sample.blur_sigma > 0.05:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(sample.blur_sigma)))
    if sample.noise_std > 0.2:
        arr = np.asarray(out, dtype=np.float32)
        noise = rng.normal(0.0, sample.noise_std, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        out = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if sample.show_through > 0.02:
        ghost = out.convert("L")
        ghost = ImageEnhance.Contrast(ghost).enhance(0.35)
        ghost = ghost.transpose(Image.FLIP_LEFT_RIGHT)
        ghost = ghost.transform(
            ghost.size,
            Image.AFFINE,
            (1, 0, 6, 0, 1, -4),
            fillcolor=255,
        )
        ghost_rgb = Image.merge("RGB", (ghost, ghost, ghost))
        out = Image.blend(out, ghost_rgb, alpha=float(np.clip(sample.show_through, 0, 1)) * 0.45)
    return out


@dataclass
class DegradationProfile:
    """
    The sampleable distribution Stage 1's acceptance criteria asked for,
    not four hardcoded constants. `sample()` draws one of the measured
    pages (empirical bootstrap), which keeps the joint structure —
    blurry pages are also often noisy — instead of independently
    sampling each margin and producing impossible combinations.
    """

    samples: list[DegradationSample]
    blur_curve: list[tuple[float, float]]

    def sample(self, rng: np.random.Generator | None = None) -> DegradationSample:
        """Draw one measured page's parameters. Empty profile → clean (all zeros)."""
        if not self.samples:
            return DegradationSample(0.0, 0.0, 0.0, 0.0, source_id="empty")
        rng = rng or np.random.default_rng()
        idx = int(rng.integers(0, len(self.samples)))
        return self.samples[idx]

    def fixed(self, which: str = "median") -> DegradationSample:
        """
        Tier A needs one frozen degradation so glyph frequency is the
        only moving part. Median of each marginal, not a real joint
        sample — that is intentional: Tier A is the controlled
        instrument, not a realistic page.
        """
        if not self.samples:
            return DegradationSample(0.0, 0.0, 0.0, 0.0, source_id=f"fixed-{which}")
        def agg(attr):
            vals = np.array([getattr(s, attr) for s in self.samples], dtype=np.float64)
            if which == "median":
                return float(np.median(vals))
            if which == "p90":
                return float(np.percentile(vals, 90))
            return float(np.mean(vals))
        return DegradationSample(
            blur_sigma=agg("blur_sigma"),
            noise_std=agg("noise_std"),
            skew_degrees=agg("skew_degrees"),
            show_through=agg("show_through"),
            source_id=f"fixed-{which}",
        )

    def to_dict(self) -> dict:
        return {
            "samples": [s.to_dict() for s in self.samples],
            "blur_curve": [list(pair) for pair in self.blur_curve],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DegradationProfile":
        samples = [DegradationSample.from_dict(s) for s in payload.get("samples", [])]
        curve = [tuple(p) for p in payload.get("blur_curve", [])]
        return cls(samples=samples, blur_curve=curve)


def save_profile(profile: DegradationProfile, path: Path | str | None = None) -> Path:
    """Checkpoint the measured distribution. Resume = load, do not re-measure."""
    path = Path(path) if path is not None else PROFILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_profile(path: Path | str | None = None) -> DegradationProfile:
    path = Path(path) if path is not None else PROFILE_PATH
    if not path.exists():
        return DegradationProfile(samples=[], blur_curve=[])
    return DegradationProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _collect_scan_paths(
    extra_images: Sequence[Path | str] | None = None,
    include_glotocr_degraded: bool = True,
    max_glotocr: int = 12,
) -> list[Path]:
    """
    Prefer real scans in SCAN_DIR (hi-res IA pages, later: phone photos
    of forms). Always mix in a capped GlotOCR `*_degraded.png` tail —
    well-scanned DLI books at JPEG-IIIF often have near-zero paper
    noise and show-through, and Tier B still needs that heavier mode.
    Never use the 700px layout-bank thumbnails: bilinear on those
    thumbnails would measure our downsample, not the scanner.
    """
    paths: list[Path] = []
    if SCAN_DIR.exists():
        for p in sorted(SCAN_DIR.glob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                paths.append(p)
    if extra_images:
        paths.extend(Path(p) for p in extra_images)
    if include_glotocr_degraded:
        glot: list[Path] = []
        for folder in GLOTOCR_IMAGE_DIRS:
            if folder.exists():
                glot.extend(sorted(folder.glob("*_degraded.png")))
        paths.extend(glot[:max_glotocr])
    seen = set()
    unique = []
    for p in paths:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(p)
    return unique


def _sharp_gray_for_calibration() -> np.ndarray:
    """
    A born-digital full page at ~120 dpi, so the blur curve lives in
    the same pixel scale as the 1600px-wide IA measurement scans.
    Line-level GlotOCR `*_plain.png` images are the wrong scale (too
    few pixels of edge) and made the first fit report sigma≈0 on every
    real book page.
    """
    try:
        import pymupdf
    except ImportError:
        pymupdf = None
    pdf_dir = REPO_ROOT / "data" / "cache" / "layouts" / "pdfs"
    if pymupdf is not None and pdf_dir.exists():
        pdfs = sorted(pdf_dir.glob("wiki*.pdf"))
        if pdfs:
            doc = pymupdf.open(pdfs[0])
            try:
                page = doc[min(1, len(doc) - 1)]
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(120 / 72, 120 / 72),
                    colorspace=pymupdf.csGRAY,
                )
                return np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width
                ).astype(np.float32)
            finally:
                doc.close()
    sharp_paths = _collect_sharp_paths()
    if sharp_paths:
        with Image.open(sharp_paths[0]) as im:
            return _as_gray(im)
    synth = np.full((200, 400), 240.0, dtype=np.float32)
    synth[40:160, 40:80] = 20.0
    synth[40:160, 120:160] = 20.0
    return synth


def fetch_measurement_scans(n_per_item: int = 10, width: int = 1600) -> list[Path]:
    """
    Pull IA pages at measurement resolution into SCAN_DIR.

    Layout extraction uses ~800px thumbnails; blur/noise do not survive
    that downsample, so this is a separate fetch, cached independently.
    """
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for ia_id in INTERNET_ARCHIVE_IDS:
        out.extend(
            fetch_internet_archive_pages(
                ia_id, n_pages=n_per_item, width=width, dest_dir=SCAN_DIR
            )
        )
    return out


def _collect_sharp_paths() -> list[Path]:
    paths = []
    for folder in GLOTOCR_IMAGE_DIRS:
        if folder.exists():
            paths.extend(sorted(folder.glob("*_plain.png"))[:4])
    return paths


def fit_profile(
    extra_images: Sequence[Path | str] | None = None,
    include_glotocr_degraded: bool = True,
    fetch: bool = True,
) -> DegradationProfile:
    """
    Measure every available scan, write `profile.json`, return the
    distribution. Safe to re-run — measurement is deterministic given
    the same files. Called once per machine; `render.py` only loads.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_DIR.mkdir(parents=True, exist_ok=True)
    if fetch:
        fetch_measurement_scans()

    curve = build_blur_calibration(_sharp_gray_for_calibration())

    samples = []
    for path in _collect_scan_paths(extra_images, include_glotocr_degraded):
        with Image.open(path) as im:
            samples.append(measure_image(im, curve, source_id=str(path.name)))
    profile = DegradationProfile(samples=samples, blur_curve=curve)
    save_profile(profile)
    return profile


def main() -> None:
    """CLI: measure whatever scans are on disk and print the distribution summary."""
    profile = fit_profile()
    print(f"wrote {PROFILE_PATH} ({len(profile.samples)} samples)")
    if not profile.samples:
        print("  (empty — drop scans in data/cache/degradation/scans/)")
        return
    for attr in ("blur_sigma", "noise_std", "skew_degrees", "show_through"):
        vals = np.array([getattr(s, attr) for s in profile.samples])
        print(f"  {attr}: median={np.median(vals):.3f}  p10={np.percentile(vals, 10):.3f}  "
              f"p90={np.percentile(vals, 90):.3f}")


if __name__ == "__main__":
    main()

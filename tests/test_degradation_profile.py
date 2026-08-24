"""Checkable properties of degradation measurement: invert known damage."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renderer.degradation_profile import (  # noqa: E402
    DegradationProfile,
    DegradationSample,
    apply_degradation,
    build_blur_calibration,
    estimate_blur_sigma,
    estimate_noise_std,
    estimate_skew_degrees,
    estimate_show_through,
    measure_image,
    save_profile,
    load_profile,
)


def _sharp_page(w=240, h=320) -> np.ndarray:
    """High-contrast fake text lines — enough edge energy to measure."""
    im = Image.new("L", (w, h), 245)
    draw = ImageDraw.Draw(im)
    for y in range(30, 290, 18):
        draw.rectangle([20, y, 220, y + 8], fill=15)
    return np.asarray(im, dtype=np.float32)


class BlurInversionTests(unittest.TestCase):
    def test_known_sigma_recovers(self):
        sharp = _sharp_page()
        curve = build_blur_calibration(sharp)
        pil = Image.fromarray(sharp.astype(np.uint8), mode="L")
        for true_sigma in (0.8, 1.8, 3.5):
            blurred = np.asarray(
                pil.filter(ImageFilter.GaussianBlur(radius=true_sigma)), dtype=np.float32
            )
            est = estimate_blur_sigma(blurred, curve)
            self.assertLess(
                abs(est - true_sigma), 0.6,
                msg=f"sigma {true_sigma} recovered as {est}",
            )

    def test_sharp_image_is_near_zero(self):
        sharp = _sharp_page()
        curve = build_blur_calibration(sharp)
        self.assertLess(estimate_blur_sigma(sharp, curve), 0.35)


class SkewTests(unittest.TestCase):
    def test_rotated_lines_recover_sign_and_scale(self):
        sharp = Image.fromarray(_sharp_page().astype(np.uint8), mode="L")
        for true_angle in (-3.0, 2.5):
            rot = sharp.rotate(true_angle, resample=Image.BILINEAR, fillcolor=255)
            est = estimate_skew_degrees(np.asarray(rot, dtype=np.float32), search=6.0, step=0.5)
            self.assertGreater(est * true_angle, 0, msg=f"sign mismatch {true_angle} vs {est}")
            self.assertLess(abs(est - true_angle), 1.5)


class NoiseTests(unittest.TestCase):
    def test_added_noise_raises_estimate(self):
        sharp = _sharp_page()
        clean = estimate_noise_std(sharp)
        rng = np.random.default_rng(0)
        noisy = np.clip(sharp + rng.normal(0, 12, sharp.shape), 0, 255)
        self.assertGreater(estimate_noise_std(noisy), clean + 3)


class ShowThroughTests(unittest.TestCase):
    def test_ghosted_background_raises_estimate(self):
        sharp = _sharp_page()
        clean = estimate_show_through(sharp)
        ghost = np.clip(np.roll(_sharp_page(), 9, axis=0) * 0.35 + 160, 0, 255)
        mixed = np.clip(0.75 * sharp + 0.25 * ghost, 0, 255)
        self.assertGreater(estimate_show_through(mixed), clean)


class ApplyAndProfileTests(unittest.TestCase):
    def test_apply_changes_pixels(self):
        im = Image.fromarray(_sharp_page().astype(np.uint8), mode="L").convert("RGB")
        sample = DegradationSample(blur_sigma=2.0, noise_std=8.0, skew_degrees=2.0, show_through=0.4)
        out = apply_degradation(im, sample, rng=np.random.default_rng(1))
        self.assertEqual(out.size, im.size)
        self.assertGreater(np.mean(np.abs(np.asarray(out).astype(float) - np.asarray(im.convert("RGB")))), 1.0)

    def test_profile_sample_and_roundtrip(self):
        samples = [
            DegradationSample(1.0, 5.0, 0.5, 0.1, "a"),
            DegradationSample(2.0, 9.0, -1.0, 0.3, "b"),
        ]
        profile = DegradationProfile(samples=samples, blur_curve=[(0.0, 100.0), (2.0, 10.0)])
        drawn = profile.sample(np.random.default_rng(0))
        self.assertIn(drawn.source_id, {"a", "b"})
        fixed = profile.fixed("median")
        self.assertAlmostEqual(fixed.blur_sigma, 1.5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            save_profile(profile, path)
            loaded = load_profile(path)
            self.assertEqual(len(loaded.samples), 2)
            self.assertEqual(loaded.samples[1].source_id, "b")

    def test_measure_image_returns_finite(self):
        sharp = _sharp_page()
        curve = build_blur_calibration(sharp)
        sample = measure_image(sharp, curve, source_id="unit")
        for attr in ("blur_sigma", "noise_std", "skew_degrees", "show_through"):
            val = getattr(sample, attr)
            self.assertTrue(np.isfinite(val), msg=attr)


if __name__ == "__main__":
    unittest.main()

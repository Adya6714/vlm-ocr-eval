"""
Generates tiny fake line-crop images + manifests, purely for smoke-
testing the instrument pipeline (tokenizer, encoder, decoder, train.py,
probe1_exposure.py) without needing Stage 1's real renderer output.

Noise line-crops + manifest — not a finding. Images are random noise,
not rendered text. This proves the training/generation/checkpointing
machinery works; it does not and cannot produce a real finding (see
Makefile's probe1-smoke target, which prints this caveat before
running). One manifest per glyph-frequency condition (natural /
flattened / inverted), matching the real pipeline's one-manifest-
per-condition shape reused across that condition's three seeds.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

OUTPUT_DIR = "/tmp/fake_lines"
CONDITIONS = ("natural", "flattened", "inverted")

# Small Devanagari strings; each string is repeated enough times that
# every grapheme cluster clears train.py's tokenizer.build_vocab(...,
# min_freq=5) floor. Sparse fake data would silently shrink the vocab
# to specials-only rather than failing loudly.
FAKE_TEXTS = [
    "हिन्दी एक भाषा है।",
    "यह एक वाक्य है।",
    "भाषा सीखना अच्छा है।",
    "यह भाषा है।",
    "हिन्दी भाषा अच्छा है।",
]

# Matches generate.py __main__ smoke tensor: torch.ones(1, 1, 70, 280)
IMAGE_HEIGHT = 70
IMAGE_WIDTH = 280


def build_fake_manifest(
    condition: str,
    output_dir: str = OUTPUT_DIR,
    repeats: int = 10,
) -> str:
    """
    Writes `repeats * len(FAKE_TEXTS)` fake (image, text) pairs for one
    condition and a `{condition}.jsonl` manifest pointing to them.
    Returns the manifest path. Idempotent — re-running overwrites.
    """
    condition_dir = os.path.join(output_dir, condition)
    os.makedirs(condition_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{condition}.jsonl")

    texts = FAKE_TEXTS * repeats
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            arr = (np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH) * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L")
            img_path = os.path.join(condition_dir, f"line_{i}.png")
            img.save(img_path)
            f.write(
                json.dumps({"image_path": img_path, "text": text}, ensure_ascii=False)
                + "\n"
            )

    return manifest_path


def build_all_fake_manifests(output_dir: str = OUTPUT_DIR) -> dict[str, str]:
    """One manifest per condition; paths match makefile probe1-smoke."""
    os.makedirs(output_dir, exist_ok=True)
    return {c: build_fake_manifest(c, output_dir=output_dir) for c in CONDITIONS}


if __name__ == "__main__":
    paths = build_all_fake_manifests()
    for condition, path in paths.items():
        n_lines = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"built fake manifest ({condition}): {path} ({n_lines} examples)")

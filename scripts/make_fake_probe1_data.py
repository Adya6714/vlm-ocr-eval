"""
Generates tiny fake line-crop images + a manifest, purely for smoke-
testing src/models/instrument/'s pipeline (tokenizer, encoder, decoder,
train.py, probe1_exposure.py) without needing Stage 1's real renderer
output to exist yet.

NOT real data -- images are random noise, not rendered text. This
proves the training/generation/checkpointing MACHINERY works, it does
not and cannot produce a real finding (see Makefile's probe1-smoke
target, which prints this caveat before running). Extracted into its
own script (rather than living inline in a test) specifically so
`make smoke-test` is one command, not a copy-pasted snippet someone
has to remember.
"""

import json
import os

import numpy as np
from PIL import Image

OUTPUT_DIR = "/tmp/fake_lines"

FAKE_TEXTS = [
    "हिन्दी एक भाषा है।",
    "यह एक वाक्य है।",
    "भाषा सीखना अच्छा है।",
    "यह भाषा है।",
    "हिन्दी भाषा अच्छा है।",
]


def build_fake_manifest(output_dir: str = OUTPUT_DIR, repeats: int = 8) -> str:
    """
    Writes `repeats * len(FAKE_TEXTS)` fake (image, text) pairs and a
    manifest.jsonl pointing to them. Returns the manifest path.
    Idempotent -- re-running overwrites, doesn't append/duplicate.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")

    texts = FAKE_TEXTS * repeats
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            width = 280 + (i % 5) * 42  # varied widths, multiples of 14
            height = 70
            arr = (np.random.rand(height, width) * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L")
            img_path = os.path.join(output_dir, f"line_{i}.png")
            img.save(img_path)
            f.write(json.dumps({"image_path": img_path, "text": text}, ensure_ascii=False) + "\n")

    return manifest_path


if __name__ == "__main__":
    path = build_fake_manifest()
    n_lines = sum(1 for _ in open(path, encoding="utf-8"))
    print(f"built fake manifest: {path} ({n_lines} examples)")

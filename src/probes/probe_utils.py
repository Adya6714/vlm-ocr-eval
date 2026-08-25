"""
Shared helpers for probes that load a trained Stage 2a checkpoint and
run it against real images. Extracted from probe3_blank_control.py so
probe5_calibration.py (and any later probe) don't duplicate this.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Same pattern as probe1_exposure.py / probe3_blank_control.py: this
# file lives in src/probes/, but generate/train/tokenizer live in
# src/models/instrument/. Insert that directory so bare sibling imports
# work regardless of cwd.
_INSTRUMENT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "instrument")
)
if _INSTRUMENT_DIR not in sys.path:
    sys.path.insert(0, _INSTRUMENT_DIR)

from generate import generate
from tokenizer import GraphemeTokenizer
from train import InstrumentModel, PATCH_SIZE


def load_model_and_tokenizer(output_root: Path, condition: str, seed: int, device: torch.device):
    tokenizer_path = output_root / f"tokenizer_{condition}.json"
    ckpt_path = output_root / f"checkpoint_{condition}_seed{seed}.pt"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"no tokenizer at {tokenizer_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    tokenizer = GraphemeTokenizer.load(str(tokenizer_path))
    model = InstrumentModel(vocab_size=len(tokenizer))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def resize_to_canonical_height(image: Image.Image, canonical_height: int = 70) -> Image.Image:
    """
    Real Tier C images arrive at whatever height they were scanned at
    (e.g. 254px), far taller than the 70px canonical line-crop height
    every synthetic training image uses (DECISIONS.md #44). Feeding
    un-resized real images would be out-of-distribution by construction,
    not a fair synthetic-to-real comparison. Resize to the same
    canonical height, preserving aspect ratio, first.
    """
    if image.height == canonical_height:
        return image
    scale = canonical_height / image.height
    new_width = max(1, round(image.width * scale))
    return image.resize((new_width, canonical_height), Image.LANCZOS)


def prepare_image_tensor(image: Image.Image, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    if image.mode != "L":
        image = image.convert("L")
    w, h = image.size
    pad_w = ((w + patch_size - 1) // patch_size) * patch_size - w
    pad_h = ((h + patch_size - 1) // patch_size) * patch_size - h
    arr = torch.from_numpy(np.array(image, dtype="float32") / 255.0).unsqueeze(0)
    arr = torch.nn.functional.pad(arr, (0, pad_w, 0, pad_h), value=1.0)
    return arr.unsqueeze(0)


def run_generate(model, tokenizer, image: Image.Image, device: torch.device) -> dict:
    tensor = prepare_image_tensor(image).to(device)
    return generate(model, tensor, tokenizer)

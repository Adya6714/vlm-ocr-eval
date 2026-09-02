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
from train import (
    InstrumentModel,
    PATCH_SIZE,
    checkpoint_path,
    snapshot_checkpoint_path,
    tokenizer_path,
    verify_checkpoint_matches_run,
    verify_tokenizer_matches_checkpoint,
)


def load_tokenizer(output_root: Path, script: str, condition: str) -> GraphemeTokenizer:
    """Load the condition's frozen vocabulary (shared across step snapshots)."""
    tok_path = Path(tokenizer_path(str(output_root), script, condition))
    if not tok_path.exists():
        raise FileNotFoundError(f"no tokenizer at {tok_path}")
    return GraphemeTokenizer.load(str(tok_path))


def load_model_from_checkpoint_file(
    ckpt_path: Path,
    tokenizer: GraphemeTokenizer,
    script: str,
    condition: str,
    device: torch.device,
) -> tuple[InstrumentModel, dict]:
    """
    Load weights from an explicit checkpoint path (main resume file or a
    step snapshot). Used by probe3_training_curve.py to evaluate
    multiple training steps without duplicating load/verify logic.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    model = InstrumentModel(vocab_size=len(tokenizer))
    ckpt = torch.load(ckpt_path, map_location=device)
    verify_checkpoint_matches_run(ckpt, script, condition)
    verify_tokenizer_matches_checkpoint(tokenizer, ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def load_model_and_tokenizer(
    output_root: Path, script: str, condition: str, seed: int, device: torch.device,
):
    tokenizer = load_tokenizer(output_root, script, condition)
    ckpt_path = Path(checkpoint_path(str(output_root), script, condition, seed))
    model, _ = load_model_from_checkpoint_file(
        ckpt_path, tokenizer, script, condition, device,
    )
    return model, tokenizer


def resolve_checkpoint_for_step(
    output_root: Path, script: str, condition: str, seed: int, step: int,
) -> Path:
    """
    Prefer an immutable step snapshot; fall back to the main resume
    checkpoint only when its stored step matches the requested step.
    """
    snap = Path(snapshot_checkpoint_path(str(output_root), script, condition, seed, step))
    if snap.exists():
        return snap
    main = Path(checkpoint_path(str(output_root), script, condition, seed))
    if main.exists():
        ckpt = torch.load(main, map_location="cpu")
        if ckpt.get("step") == step:
            return main
    raise FileNotFoundError(
        f"no checkpoint for step {step} under {output_root} "
        f"(looked for {snap.name} and main file with matching step). "
        f"Retrain with train.py --keep-snapshots to retain intermediate weights."
    )


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

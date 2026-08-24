"""
Scaled export driver: render N pages per (script, mode), crop to lines,
write one manifest per (script, mode).

Why this exists: `export_line_manifest.py` is the page→line-crop adapter
for a single `RenderedPage`. Probe 1 / `train.py` need hundreds of those
rows per glyph-frequency condition. This module is the batch driver on
top — walk modes, render Tier A pages, append crops, checkpoint progress
so a killed Colab session can resume (AGENTS.md Long-running scripts).

Usage:
    python src/data_pipeline/export_manifest_scaled.py \\
        --script hindi --pages-per-mode 100
    # or: python -m data_pipeline.export_manifest_scaled  (with cwd/src on path)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Same pattern as probe1_exposure.py: this file lives under
# src/data_pipeline/, but imports `renderer.*` and sibling
# `data_pipeline.*` which expect `src/` on sys.path. Inserting the
# parent of this package keeps `python path/to/this_file.py` working
# regardless of cwd, without requiring a second copy of the adapter.
_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np

from renderer.render import render_tier_a
from renderer.glyph_frequency import MODES
from data_pipeline.export_line_manifest import export_line_crops, append_manifest

# Relative to --data-root / OCR_DATA_ROOT (default data/), matching
# fetch_glotocr.py's layout under data/raw/{language}/.
SCRIPT_GT_FILES = {
    "hindi": "hindi/ground_truth.jsonl",
    "bengali": "bengali/ground_truth.jsonl",
}
SCRIPT_HB_KEY = {"hindi": "deva", "bengali": "beng"}


def resolve_data_root(data_root: str | None = None) -> Path:
    """
    Single Colab/local knob, same contract as run_baselines.py.

    Returns the data root Path (default: $OCR_DATA_ROOT or `data/`).
    Inputs live at `{root}/raw`, outputs at `{root}/manifests` and
    `{root}/cache/line_crops` — not a separate invented `--root` that
    pointed at the repo and then hard-coded `data/` underneath.
    """
    root = (
        data_root
        if data_root is not None
        else os.environ.get("OCR_DATA_ROOT", "data")
    )
    return Path(root)


def load_corpus(data_root: Path, script: str) -> list[str]:
    """
    Load ground-truth text lines for resampling into Tier A pages.

    Reads `{data_root}/raw/{script}/ground_truth.jsonl` produced by
    fetch_glotocr.py — the same corpus Stage 0 baselines already use,
    so Probe 1's glyph-frequency modes stay tied to real Indic text
    rather than invented filler.
    """
    gt_path = data_root / "raw" / SCRIPT_GT_FILES[script]
    return [
        json.loads(line)["text"]
        for line in gt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def progress_path(data_root: Path, script: str, mode: str) -> Path:
    """Per-(script, mode) resume cursor next to the manifest it guards."""
    return data_root / "manifests" / f"{script}_{mode}.progress.json"


def load_progress(path: Path) -> int:
    """Next page index to render; 0 if this mode has never been started."""
    return json.loads(path.read_text())["next_page"] if path.exists() else 0


def save_progress(path: Path, next_page: int) -> None:
    """
    Checkpoint after each page so Ctrl-C / Colab death loses at most one
    page of work. Manifest rows are already appended by then; the
    progress file is what keeps a rerun from appending duplicates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"next_page": next_page}))


def page_rng(script: str, mode: str, page_idx: int) -> np.random.Generator:
    """
    Deterministic RNG per page, stable across processes and PYTHONHASHSEED.

    Python's built-in hash() is randomized per process by default, so a
    resumed run that somehow re-rendered page_idx would not reproduce
    the same page. sha256 of a stable key keeps seeds reproducible.
    """
    key = f"{script}:{mode}:{page_idx}".encode()
    seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def run(data_root: Path, script: str, pages_per_mode: int) -> None:
    """
    For each glyph-frequency mode: render pages, crop lines, append
    manifest, advance progress. Skips modes already at pages_per_mode.
    """
    corpus = load_corpus(data_root, script)
    hb_script = SCRIPT_HB_KEY[script]
    crop_dir = data_root / "cache" / "line_crops" / script
    manifest_dir = data_root / "manifests"

    for mode in MODES:
        manifest_path = manifest_dir / f"{script}_{mode}.jsonl"
        prog_path = progress_path(data_root, script, mode)
        start_page = load_progress(prog_path)

        if start_page >= pages_per_mode:
            print(
                f"[{script}/{mode}] already complete "
                f"({start_page}/{pages_per_mode}) — skipping",
                flush=True,
            )
            continue

        print(
            f"[{script}/{mode}] resuming at page "
            f"{start_page}/{pages_per_mode}",
            flush=True,
        )
        for page_idx in range(start_page, pages_per_mode):
            t0 = time.time()
            rng = page_rng(script, mode, page_idx)
            page = render_tier_a(corpus, mode=mode, script=hb_script, rng=rng)
            rows = export_line_crops(
                page, crop_dir, stem=f"{mode}_page{page_idx:04d}"
            )
            append_manifest(rows, manifest_path)
            save_progress(prog_path, page_idx + 1)
            # Per-page progress: page render+crop is seconds-scale, so
            # silence between every-10 prints looks like a hang
            # (AGENTS.md Long-running scripts).
            print(
                f"[{script}/{mode}] page {page_idx + 1}/{pages_per_mode} "
                f"({len(rows)} lines, {time.time() - t0:.2f}s)",
                flush=True,
            )
        print(f"[{script}/{mode}] done: {manifest_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render Tier A pages and export line-crop manifests"
    )
    ap.add_argument("--script", required=True, choices=list(SCRIPT_GT_FILES))
    ap.add_argument("--pages-per-mode", type=int, default=100)
    ap.add_argument(
        "--data-root",
        dest="data_root",
        default=None,
        help=(
            "Single data root (default: $OCR_DATA_ROOT or data/). "
            "Raw GT at {root}/raw, crops at {root}/cache/line_crops, "
            "manifests at {root}/manifests."
        ),
    )
    args = ap.parse_args()
    data_root = resolve_data_root(args.data_root)
    print(f"[paths] data_root={data_root}", flush=True)
    run(data_root, args.script, args.pages_per_mode)


if __name__ == "__main__":
    main()

"""
Crop line-level images out of a rendered page and write Stage 2a's manifest.

Why this exists: `train.py` (Stage 2a) reads JSONL rows of
`{"image_path", "text"}` line crops. `render.py` paints full pages and
now records `PageGT.lines` with page-space boxes (DECISIONS.md #41).
This module is the adapter between those two shapes — crop + append —
with no shaping or painting of its own.

Called after `render_page` / `render_tier_a`. Output lands under a
`--data-root`-style tree (`data/cache/line_crops/`,
`data/manifests/...`) so Colab and the laptop share one layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from renderer.render import RenderedPage


def export_line_crops(
    page: RenderedPage,
    out_dir: Path | str,
    stem: str,
    pad: int = 2,
) -> list[dict]:
    """
    Crop each `LineGT` box from `page.image` into `{stem}_rXXXX.png`.

    Why pad=2: the LineGT vertical pad is relative to font metrics; a
    couple of extra pixels absorbs anti-alias fringe so the crop does
    not clip ink that visually belongs to the line. Returns the
    `{"image_path", "text"}` rows `train.py`'s LineDataset expects —
    callers decide whether to write them (see `append_manifest`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lg in page.ground_truth.lines:
        x0, y0, x1, y1 = lg.bbox
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(page.image.width, x1 + pad), min(page.image.height, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = page.image.crop((x0, y0, x1, y1))
        img_path = out_dir / f"{stem}_r{lg.reading_order:04d}.png"
        crop.save(img_path)
        rows.append({"image_path": str(img_path), "text": lg.text})
    return rows


def append_manifest(rows: list[dict], manifest_path: Path | str) -> None:
    """
    Append rows to a JSONL manifest, creating parent dirs if needed.

    Append (not overwrite) so a killed Colab run can resume by
    re-rendering only unfinished pages — matching AGENTS.md's default
    checkpoint style. Delete the file first for a clean rebuild.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

"""
Ensure `data/cache/line_crops/` exists for a given manifest.

Why this exists
---------------
`data/manifests/*.jsonl` are committed, but the corresponding PNG line crops under
`data/cache/line_crops/<script>/` are **gitignored** (large, regenerable).

That is fine until a training script assumes the crops are present and crashes on
the first missing file (this happened in Stage 2b demo SFT).

This helper bridges that operational gap:
- Read a manifest of `{"image_path","text"}` rows.
- Detect which page stems (e.g. `natural_page0000`) are missing any crop PNGs.
- Re-render just those missing pages deterministically (same RNG as
  `export_manifest_scaled.py`) and write the crops at the exact paths the manifest
  expects.

It is CPU-only and resumable-by-default: it skips any stem that already has all
expected files.

Called from
-----------
- Colab: before `python src/models/demo/sft.py --run ...` when `line_crops/` was not
  copied from Drive.
- Laptop: same situation if a fresh checkout has manifests but not caches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Keep `python src/data_pipeline/ensure_line_crops.py` working regardless of cwd.
_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_pipeline.export_manifest_scaled import (
    SCRIPT_HB_KEY,
    SCRIPT_GT_FILES,
    load_corpus,
    page_rng,
    resolve_data_root,
)
from renderer.render import render_tier_a
from data_pipeline.export_line_manifest import export_line_crops


_LINE_CROP_RE = re.compile(
    r"^(?P<prefix>.*?/)?(?P<script>[a-z_]+)/(?P<mode>[a-z_]+)_page(?P<page>\d{4})_r(?P<row>\d{4})\.png$"
)


@dataclass(frozen=True)
class StemKey:
    script: str
    mode: str
    page_idx: int
    stem: str  # e.g. "natural_page0000"


def _parse_stem_key(image_path: str) -> StemKey:
    """
    Parse `data/cache/line_crops/<script>/<mode>_pageXXXX_rYYYY.png`.

    We deliberately do *not* accept arbitrary filenames: if a manifest row does not
    follow the project's canonical naming, this script refuses to guess.
    """
    m = _LINE_CROP_RE.match(image_path.replace("\\", "/"))
    if not m:
        raise ValueError(
            f"unrecognized line-crop path {image_path!r}; expected "
            "'data/cache/line_crops/<script>/<mode>_page0000_r0000.png'"
        )
    script = m.group("script")
    mode = m.group("mode")
    page_idx = int(m.group("page"))
    stem = f"{mode}_page{page_idx:04d}"
    return StemKey(script=script, mode=mode, page_idx=page_idx, stem=stem)


def _resolve_manifest_paths(manifest: Path, data_root: Path) -> list[Path]:
    """
    Resolve image paths the same way `LineCropDataset` does (demo SFT).

    - If `image_path` is relative: treat as relative to `data_root/` first.
    - If still missing: treat as relative to `data_root.parent/` (repo root).
    """
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths: list[Path] = []
    for r in rows:
        p = Path(r["image_path"])
        if p.is_absolute():
            paths.append(p)
            continue
        cand = data_root / p
        if not cand.exists():
            cand = data_root.parent / p
        paths.append(cand)
    return paths


def ensure_line_crops(
    *,
    repo_root: Path,
    data_root: Path,
    manifest_path: Path,
    script: str,
    verify_only: bool,
    limit_pages: int | None,
) -> int:
    """
    Ensure all PNGs referenced by `manifest_path` exist on disk.

    Returns an exit code: 0 if all crops exist (or were generated), 2 if missing and
    verify_only, 1 on other errors.
    """
    if script not in SCRIPT_GT_FILES:
        raise SystemExit(
            f"unknown script {script!r}; expected one of {sorted(SCRIPT_GT_FILES)}"
        )
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    # Group required rows by (script, mode, page_idx) stem.
    required_by_stem: dict[StemKey, set[Path]] = defaultdict(set)
    resolved_paths = _resolve_manifest_paths(manifest_path, data_root)
    for p in resolved_paths:
        # Parse using manifest-relative string, not resolved absolute path.
        rel = os.path.relpath(p, repo_root).replace("\\", "/")
        key = _parse_stem_key(rel)
        if key.script != script:
            raise SystemExit(
                f"manifest contains script {key.script!r}, but --script is {script!r}"
            )
        required_by_stem[key].add(p)

    stems = sorted(required_by_stem.keys(), key=lambda k: (k.mode, k.page_idx))
    if limit_pages is not None:
        stems = stems[:limit_pages]

    missing_stems: list[StemKey] = []
    missing_files = 0
    for k in stems:
        missing = [p for p in required_by_stem[k] if not p.exists()]
        if missing:
            missing_stems.append(k)
            missing_files += len(missing)

    print(f"[ensure_line_crops] manifest={manifest_path}", flush=True)
    print(f"[ensure_line_crops] stems={len(stems)} missing_stems={len(missing_stems)} missing_files={missing_files}", flush=True)

    if not missing_stems:
        print("[ensure_line_crops] OK: all crop files exist", flush=True)
        return 0
    if verify_only:
        print(
            "[ensure_line_crops] verify-only: missing crops found; refusing to render.",
            flush=True,
        )
        return 2

    # Load corpus text once. This requires `{data_root}/raw/<script>/ground_truth.jsonl`.
    corpus = load_corpus(data_root, script)
    hb_script = SCRIPT_HB_KEY[script]

    crop_dir = data_root / "cache" / "line_crops" / script
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Render just the missing stems; skip if another process filled them mid-run.
    for i, k in enumerate(missing_stems, start=1):
        # If they all exist now, skip.
        still_missing = [p for p in required_by_stem[k] if not p.exists()]
        if not still_missing:
            print(f"[ensure_line_crops] {k.stem}: already satisfied — skipping", flush=True)
            continue

        t0 = time.time()
        rng = page_rng(script, k.mode, k.page_idx)
        page = render_tier_a(corpus, mode=k.mode, script=hb_script, rng=rng)
        export_line_crops(page, crop_dir, stem=k.stem)

        dt = time.time() - t0
        after_missing = [p for p in required_by_stem[k] if not p.exists()]
        if after_missing:
            # Loud failure: we must not pretend we fixed it.
            print(f"[ensure_line_crops] ERROR: {k.stem} still missing {len(after_missing)} files after render", flush=True)
            for p in sorted(after_missing)[:5]:
                print(f"  missing: {p}", flush=True)
            return 1
        print(f"[ensure_line_crops] rendered {k.stem} ({i}/{len(missing_stems)}) in {dt:.2f}s", flush=True)

    print("[ensure_line_crops] DONE: all missing stems rendered", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensure cache line crops referenced by a manifest exist")
    ap.add_argument("--script", required=True, choices=sorted(SCRIPT_GT_FILES))
    ap.add_argument("--manifest", required=True, help="Manifest JSONL (committed) that references line crop PNGs")
    ap.add_argument(
        "--data-root",
        default=None,
        help="Data root (default: $OCR_DATA_ROOT or data/). Uses {root}/raw for corpus and {root}/cache/line_crops for output.",
    )
    ap.add_argument("--verify-only", action="store_true", help="Only report missing; do not render")
    ap.add_argument("--limit-pages", type=int, default=None, help="Only check/render first N page stems (debug)")
    args = ap.parse_args()

    data_root = resolve_data_root(args.data_root)
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    return ensure_line_crops(
        repo_root=repo_root,
        data_root=data_root,
        manifest_path=manifest_path,
        script=args.script,
        verify_only=args.verify_only,
        limit_pages=args.limit_pages,
    )


if __name__ == "__main__":
    raise SystemExit(main())


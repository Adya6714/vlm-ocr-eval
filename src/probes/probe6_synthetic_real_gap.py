"""
Probe 6 (paper-deadline scope) — synthetic Claim B vs real Tier C.

Claim B (confidence ≈ as high on blank as on real text) was measured
on synthetic line crops from data/manifests/hindi_*.jsonl (Probe 3/5).
The obvious objection: does that pattern survive on real GlotOCR Tier C
scans the instrument never trained on?

This module:
  1. Asserts no path overlap between training manifests and
     data/raw/hindi/images/ (held-out validity).
  2. Runs generate() on every Hindi GT row's plain + degraded image,
     plus a blank control sized from the plain crop (probe3 make_blank).
  3. Records mean_confidence and Tier 1/2 line accuracy (same
     is_correct as probe5_calibration.py).

Full original Probe 6 (Tier B degradation sweep, handwriting anecdote,
held-out synthetic pages 100–109 per DECISIONS.md #45, multi-system
gaps) is explicitly deferred — see DECISIONS.md #58 and the analysis
doc's Future work section.

Outputs: data/probe_results/probe6_synthetic_real_hindi_seed{N}.jsonl
Analysis: src/analysis/analyze_probe6.py →
docs/probe6_synthetic_real_analysis.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_PROBES_DIR = Path(__file__).resolve().parent
if str(_PROBES_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBES_DIR))

_EVAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "eval")
)
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from equivalence_tables import tier1_equivalent  # noqa: E402
from transliteration_equivalence import tier2_equivalent, SCRIPT_MAP  # noqa: E402

from probe3_blank_control import make_blank  # noqa: E402
from probe5b_zeroshot_floor import (  # noqa: E402
    load_ground_truth_rows,
    resolve_repo_root,
)
from probe_utils import (  # noqa: E402
    load_model_and_tokenizer,
    resize_to_canonical_height,
    run_generate,
)
from train import checkpoint_path  # noqa: E402

CONDITIONS = ("real_plain", "real_degraded", "blank")
LANGUAGE = "hindi"


def is_correct(prediction: str, ground_truth: str, language: str) -> bool:
    """
    Same correctness gate as probe5_calibration.py — exact / Tier 1,
    then Tier 2 only for SCRIPT_MAP languages. Shared so Probe 6 real
    accuracy is comparable to Probe 5 synthetic accuracy.
    """
    if tier1_equivalent(prediction, ground_truth):
        return True
    if language in SCRIPT_MAP:
        return tier2_equivalent(ground_truth, prediction, language)
    return False


def collect_manifest_image_paths(manifest_path: Path) -> set[str]:
    """All image_path values from a line-crop training manifest."""
    paths: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        p = row.get("image_path")
        if p:
            paths.add(p)
    return paths


def collect_raw_image_paths(raw_images_dir: Path) -> set[str]:
    """Basenames + repo-relative paths under data/raw/hindi/images/."""
    if not raw_images_dir.is_dir():
        raise FileNotFoundError(f"no raw images dir at {raw_images_dir}")
    out: set[str] = set()
    for p in raw_images_dir.iterdir():
        if p.is_file():
            out.add(p.name)
            out.add(str(p))
            # Common repo-relative form used in ground_truth.jsonl.
            out.add(f"data/raw/hindi/images/{p.name}")
    return out


def assert_no_train_raw_leakage(
    manifests_dir: Path,
    raw_images_dir: Path,
    script: str = "hindi",
) -> dict:
    """
    Confirm training manifests do not point at Tier C GlotOCR files.

    Why this exists: Probe 6 is only a valid held-out test if the
    instrument never trained on data/raw/hindi/images/. Manifests are
    renderer line crops under data/cache/line_crops/; raw images come
    from fetch_glotocr.py. Overlap would invalidate the claim.

    Called from: run_probe6 (before any inference) and unit tests.
    Returns a summary dict written into the analysis doc.
    """
    raw_set = collect_raw_image_paths(raw_images_dir)
    overlaps: list[dict] = []
    per_manifest: dict[str, int] = {}

    for cond in ("natural", "flattened", "inverted"):
        man = manifests_dir / f"{script}_{cond}.jsonl"
        if not man.exists():
            per_manifest[man.name] = -1  # missing
            continue
        man_paths = collect_manifest_image_paths(man)
        per_manifest[man.name] = len(man_paths)
        for p in man_paths:
            name = Path(p).name
            # Path substring check: any training path under raw/hindi.
            norm = p.replace("\\", "/")
            if "raw/hindi" in norm or name in raw_set or p in raw_set:
                overlaps.append({"manifest": man.name, "image_path": p})

    summary = {
        "manifests_dir": str(manifests_dir),
        "raw_images_dir": str(raw_images_dir),
        "n_raw_images": len([p for p in raw_images_dir.iterdir() if p.is_file()]),
        "per_manifest_n_paths": per_manifest,
        "n_overlaps": len(overlaps),
        "overlaps": overlaps[:20],  # cap for readability
        "leakage_free": len(overlaps) == 0,
    }
    if overlaps:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(overlaps)} training manifest path(s) "
            f"overlap with {raw_images_dir}. First: {overlaps[0]}"
        )
    return summary


def load_completed_keys(out_path: Path) -> set[tuple[str, str]]:
    """Resume key: (condition, image_path)."""
    if not out_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["condition"], row["image_path"]))
    return done


def resolve_gt_image(rel: str, repo_root: Path) -> Path:
    """Resolve a ground_truth.jsonl image path (plain or degraded)."""
    candidate = Path(rel)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = repo_root / rel
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"image not found: {resolved}")


def build_tasks(data_root: Path, repo_root: Path) -> list[dict]:
    """
    One task per (GT row × condition): plain, degraded, blank.

    Blank is derived from the plain image's size (probe3 pattern) so
    real-domain blank confidence is comparable to synthetic-domain blank
    from Probe 3.
    """
    rows = load_ground_truth_rows(data_root, LANGUAGE)
    tasks: list[dict] = []
    for row in rows:
        plain_rel = row.get("img_plain_path")
        deg_rel = row.get("img_degraded_path")
        if not plain_rel or not deg_rel:
            raise KeyError(
                f"GT row missing img_plain_path/img_degraded_path: {row.get('id')}"
            )
        plain = str(resolve_gt_image(plain_rel, repo_root))
        degraded = str(resolve_gt_image(deg_rel, repo_root))
        tasks.append({
            "condition": "real_plain",
            "row": row,
            "image_path": plain,
            "source_path": plain,
        })
        tasks.append({
            "condition": "real_degraded",
            "row": row,
            "image_path": degraded,
            "source_path": degraded,
        })
        tasks.append({
            "condition": "blank",
            "row": row,
            "image_path": plain,  # resume key; pixels are blanked
            "source_path": plain,
            "blank_source_path": plain,
        })
    return tasks


def run_probe6(
    output_root: Path,
    data_root: Path,
    script: str,
    condition: str,
    seed: int,
    out_path: Path,
    device_str: str = "cpu",
    manifests_dir: Path | None = None,
) -> dict:
    """
    Inference pass for one hindi/natural checkpoint on Tier C + blank.

    Runs leakage assert first. Synthetic comparison numbers come from
    existing probe3/probe5 jsonl at analysis time — this script only
    writes the real-domain side.
    """
    manifests_dir = manifests_dir or (data_root / "manifests")
    raw_images = data_root / "raw" / LANGUAGE / "images"
    leakage = assert_no_train_raw_leakage(manifests_dir, raw_images, script)
    print(
        f"[probe6] leakage check OK — 0 overlaps "
        f"(manifests={leakage['per_manifest_n_paths']}, "
        f"raw_images={leakage['n_raw_images']})"
    )

    device = torch.device(device_str)
    ckpt = checkpoint_path(str(output_root), script, condition, seed)
    print(f"[probe6] checkpoint: {ckpt}")
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"missing checkpoint at {ckpt} (DECISIONS.md #47 script-scoped name)"
        )

    model, tokenizer = load_model_and_tokenizer(
        output_root, script, condition, seed, device,
    )

    repo_root = resolve_repo_root(data_root)
    tasks = build_tasks(data_root, repo_root)
    completed = load_completed_keys(out_path)
    pending = [t for t in tasks if (t["condition"], t["image_path"]) not in completed]

    total = len(tasks)
    already = total - len(pending)
    print(
        f"[probe6] {script}/{condition}/seed={seed}: "
        f"{total} tasks ({already} done, {len(pending)} remaining)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist leakage summary once as a sidecar so analysis can quote it
    # without re-walking manifests.
    leak_path = out_path.with_suffix(".leakage.json")
    leak_path.write_text(json.dumps(leakage, indent=2), encoding="utf-8")

    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            row = task["row"]
            cond = task["condition"]
            if cond == "blank":
                base = resize_to_canonical_height(Image.open(task["blank_source_path"]))
                image = make_blank(base)
            else:
                image = resize_to_canonical_height(Image.open(task["source_path"]))

            out = run_generate(model, tokenizer, image, device)
            text = out["text"]
            step_confs = out["step_confidences"]
            mean_conf = float(np.mean(step_confs)) if step_confs else None
            # Blank has no meaningful accuracy against the line GT —
            # still compute for logging but flag correct=None for blank.
            if cond == "blank":
                correct = None
            else:
                correct = is_correct(text, row.get("text") or "", LANGUAGE)

            record = {
                "condition": cond,
                "checkpoint_script": script,
                "training_condition": condition,
                "seed": seed,
                "image_path": task["image_path"],
                "image_id": row.get("id"),
                "ground_truth": row.get("text"),
                "ground_truth_script": row.get("script"),
                "text": text,
                "mean_confidence": mean_conf,
                "step_confidences": step_confs,
                "correct": correct,
                "checkpoint_path": ckpt,
                "leakage_free": True,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            done = already + i + 1
            corr_str = (
                "n/a" if correct is None else ("yes" if correct else "no")
            )
            conf_str = f"{mean_conf:.4f}" if mean_conf is not None else "n/a"
            print(
                f"[probe6] {done}/{total} {cond:14s} "
                f"id={row.get('id')}  conf={conf_str}  correct={corr_str}"
            )

    print(f"[probe6] wrote {out_path}")
    return leakage


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe 6 (scoped): Tier C real vs synthetic Claim B",
    )
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument(
        "--condition", default="natural", choices=["natural", "flattened", "inverted"],
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OCR_DATA_ROOT", "data"),
    )
    ap.add_argument(
        "--manifests-dir",
        default=None,
        help="Defaults to {data-root}/manifests",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--leakage-check-only",
        action="store_true",
        help="Run the train/raw overlap assert and exit (no GPU needed)",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    manifests_dir = Path(args.manifests_dir) if args.manifests_dir else data_root / "manifests"

    if args.leakage_check_only:
        summary = assert_no_train_raw_leakage(
            manifests_dir, data_root / "raw" / LANGUAGE / "images", args.script,
        )
        print(json.dumps(summary, indent=2))
        return

    run_probe6(
        Path(args.output_root),
        data_root,
        args.script,
        args.condition,
        args.seed,
        Path(args.out),
        args.device,
        manifests_dir,
    )


if __name__ == "__main__":
    main()

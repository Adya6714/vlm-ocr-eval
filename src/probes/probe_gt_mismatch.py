"""
Shuffled image–text pairing: teacher-forced log p(GT) on *mismatched* real images.

Why this exists: blank ≈ real on mean log p(GT) is weak evidence because
blank is a degenerate input (and blank generations are highly
mode-collapsed). Pairing each ground-truth string with a *different*
real image removes that confound: if mean log p(GT | wrong real image)
≈ mean log p(GT | matched image) ≈ −1.78, the decoder is not using the
pixels to score the text.

Derangement: a fixed-seed permutation of the Probe 5b Hindi sample with
no fixed points, so every image is scored against another image's GT.
Resume key: (condition=mismatch, image_path, paired_image_id).

Colab-only heavy run. Checkpoint/resume + per-image progress required.
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

from probe_gt_likelihood import (  # noqa: E402
    gt_force_ids,
    score_one,
)
from probe5b_zeroshot_floor import resolve_repo_root  # noqa: E402
from probe_attention_ablation import build_hindi_sample  # noqa: E402
from probe_utils import (  # noqa: E402
    load_model_and_tokenizer,
    resize_to_canonical_height,
)

_INSTRUMENT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "instrument")
)
if _INSTRUMENT_DIR not in sys.path:
    sys.path.insert(0, _INSTRUMENT_DIR)

from train import checkpoint_path  # noqa: E402


def derangement_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random permutation with no fixed points.

    Rejection sampling is fine at n=60; expected trials are e≈2.7.
    """
    if n < 2:
        raise ValueError("derangement needs n≥2")
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def load_completed(out_path: Path) -> set[tuple[str, str, str]]:
    """Resume on (condition, image_path, paired_image_id)."""
    if not out_path.exists():
        return set()
    done: set[tuple[str, str, str]] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add(
            (
                row["condition"],
                row["image_path"],
                str(row["paired_image_id"]),
            )
        )
    return done


def build_mismatch_tasks(
    data_root: Path, repo_root: Path, n_samples: int, derange_seed: int
) -> list[dict]:
    """
    One mismatch task per Hindi sample image.

    paired_gt is the text from another image in the same draw; image
    pixels stay matched to image_path.
    """
    hindi = build_hindi_sample(data_root, repo_root, n_samples)
    n = len(hindi)
    perm = derangement_indices(n, np.random.default_rng(derange_seed))
    tasks: list[dict] = []
    for i, item in enumerate(hindi):
        j = int(perm[i])
        partner = hindi[j]
        tasks.append({
            "condition": "mismatch",
            "row": item["row"],
            "image_path": item["image_path"],
            "source_path": item["image_path"],
            "paired_image_id": partner["row"].get("id"),
            "paired_image_path": partner["image_path"],
            "paired_ground_truth": partner["row"].get("text") or "",
            "matched_ground_truth": item["row"].get("text") or "",
        })
    return tasks


def run_probe(
    output_root: Path,
    data_root: Path,
    script: str,
    condition: str,
    seed: int,
    n_samples: int,
    out_path: Path,
    device_str: str = "cpu",
    derange_seed: int = 0,
    dry_run: bool = False,
) -> None:
    repo_root = resolve_repo_root(data_root)
    tasks = build_mismatch_tasks(data_root, repo_root, n_samples, derange_seed)
    ckpt = checkpoint_path(str(output_root), script, condition, seed)

    if dry_run:
        done = load_completed(out_path)
        pending = [
            t
            for t in tasks
            if (t["condition"], t["image_path"], str(t["paired_image_id"])) not in done
        ]
        print(f"[gt_mismatch] DRY-RUN seed={seed} derange_seed={derange_seed}")
        print(f"  checkpoint: {ckpt} exists={Path(ckpt).exists()}")
        print(f"  tasks={len(tasks)} pending={len(pending)} out={out_path}")
        for t in pending[:3]:
            print(
                f"  sample: img={t['row'].get('id')} ← GT from "
                f"{t['paired_image_id']}"
            )
        return

    device = torch.device(device_str)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"missing checkpoint at {ckpt}")
    model, tokenizer = load_model_and_tokenizer(
        output_root, script, condition, seed, device
    )

    completed = load_completed(out_path)
    pending = [
        t
        for t in tasks
        if (t["condition"], t["image_path"], str(t["paired_image_id"])) not in completed
    ]
    total = len(tasks)
    already = total - len(pending)
    print(
        f"[gt_mismatch] {script}/{condition}/seed={seed}: "
        f"{total} tasks ({already} done, {len(pending)} remaining)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            image = resize_to_canonical_height(Image.open(task["source_path"]))
            gt = task["paired_ground_truth"]
            body = score_one(model, tokenizer, image, gt, device)
            # Also echo whether any forced id is RARE (already in body)
            record = {
                "checkpoint_script": script,
                "training_condition": condition,
                "seed": seed,
                "derange_seed": derange_seed,
                "condition": "mismatch",
                "image_path": task["image_path"],
                "image_id": task["row"].get("id"),
                "paired_image_id": task["paired_image_id"],
                "paired_image_path": task["paired_image_path"],
                "ground_truth": gt,
                "matched_ground_truth": task["matched_ground_truth"],
                "estimator": "teacher_forced_gt_loglik_mismatched_image",
                "checkpoint_path": ckpt,
                "n_forced_ids": len(gt_force_ids(tokenizer, gt)),
                **body,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done = already + i + 1
            lp = body["mean_log_p_gt"]
            lp_str = f"{lp:.4f}" if lp is not None else "n/a"
            print(
                f"[gt_mismatch] {done}/{total} "
                f"img={task['row'].get('id')}←{task['paired_image_id']} "
                f"mean_log_p={lp_str}"
            )

    lps = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("mean_log_p_gt") is not None:
            lps.append(rec["mean_log_p_gt"])
    print(
        f"[gt_mismatch] wrote {out_path}  "
        f"n={len(lps)} mean_log_p_gt={float(np.mean(lps)) if lps else 'n/a'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script", default="hindi", choices=["hindi", "bengali"])
    ap.add_argument(
        "--condition",
        default="natural",
        choices=["natural", "flattened", "inverted"],
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--derange-seed", type=int, default=0)
    ap.add_argument("--output-root", required=True)
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OCR_DATA_ROOT", "data"),
    )
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_probe(
        Path(args.output_root),
        Path(args.data_root),
        args.script,
        args.condition,
        args.seed,
        args.n_samples,
        Path(args.out),
        args.device,
        args.derange_seed,
        args.dry_run,
    )


if __name__ == "__main__":
    main()

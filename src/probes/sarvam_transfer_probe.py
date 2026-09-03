"""
src/probes/sarvam_transfer_probe.py

Stage 5 (minimal paper scope) — Sarvam confidence on zero-exposure scripts.

Why this probe exists and where it fits
----------------------------------------
Claim B in this project ("confidence does not track image readability")
was established on a small, from-scratch instrument.  A critic can fairly
ask: is this an artifact of using an undertrained toy model, or is it a
real structural property that also shows up in a production system?

This probe sends the same three-script comparison (Hindi / Santhali /
Kashmiri) to Sarvam's Doc-AI Extract endpoint and records Sarvam's
confidence per image.  The key comparison:

  Sarvam's published word accuracy:
    Hindi     95.91 %   (high accuracy expected)
    Santhali  80.32 %   (lower — Ol Chiki, low-resource)
    Kashmiri  55.93 %   (much lower — Perso-Arabic, hardest in the set)
  (Source: sarvam.ai/blogs/sarvam-vision, verified Sept 2026)

If Sarvam's confidence *does* drop in proportion to that accuracy gap,
that is a positive finding: production confidence tracks hardness.
If Sarvam's confidence stays high across all three, it replicates this
project's own Claim B on a real production model.
Report honestly either way.

Budget discipline (DECISIONS.md #19)
--------------------------------------
This probe is budgeted for:
  10 Hindi + 10 Santhali + 10 Kashmiri + 5 blank = 35 images
  @ ₹0.5 / page (confirmed docs.sarvam.ai, Sept 2026) = ₹17.50
  Under the ~₹100 / 200-page project budget, leaves substantial reserve.

DO NOT RUN THIS YOURSELF.  The probe writes to data/cache/sarvam/,
and once an image is cached it will never call the API again.  But the
first run costs real money.  Tell the user the exact command and cost
before they run it.

Outputs
-------
  data/probe_results/sarvam_transfer_probe.jsonl   — one record per image
  docs/sarvam_transfer_analysis.md                 — written by analyze step

Run command (user runs this, not the agent):
  export SARVAM_API_KEY=<your_key>
  python src/probes/sarvam_transfer_probe.py --data-root data --out data/probe_results/sarvam_transfer_probe.jsonl

Handles resume: images already in data/probe_results/sarvam_transfer_probe.jsonl
are skipped (their cache entry will be used next time anyway, but skipping
keeps incremental output correct if the script is interrupted mid-run).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from io import BytesIO
from pathlib import Path

# PIL for building blank images to send as the blank control.
# The same concept as probe3_blank_control.py: a pure-white image of
# a representative size, giving the API a valid image with no text signal.
from PIL import Image

# sarvam_client is in src/eval/ — either on PYTHONPATH or invoked with
# PYTHONPATH=src or via the run command documented above.
from eval.sarvam_client import SarvamClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sarvam Vision published accuracy (verified Sept 2026 from sarvam.ai/blogs/sarvam-vision)
# These are word-accuracy figures (100 × (1 − WER)) on Sarvam Indic OCR Bench.
# ---------------------------------------------------------------------------
SARVAM_PUBLISHED_ACCURACY = {
    "hindi": 95.91,
    "santhali": 80.32,
    "kashmiri": 55.93,
}

# BCP-47 language codes for the Extract API
LANGUAGE_CODES = {
    "hindi": "hi-IN",
    "santhali": "sat-IN",   # Santhali / Ol Chiki
    "kashmiri": "ks-Arab",  # Kashmiri (Perso-Arabic script)
}

# How many plain images to sample per script
N_PER_SCRIPT = 10
# How many blank images to send as the blank control
N_BLANK = 5
# Size for blank images (matches typical GlotOCR-bench line image width)
BLANK_SIZE = (1200, 80)


def load_ground_truth(gt_path: Path) -> list[dict]:
    """
    Load a ground_truth.jsonl into a list of dicts.

    Each row has: id, text, language, script, source,
                  img_plain_path, img_degraded_path.
    """
    rows = []
    with gt_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sample(data_root: Path, seed: int = 0) -> list[dict]:
    """
    Build the list of (script, image_path, ground_truth_text) tasks.

    Samples N_PER_SCRIPT images from each of hindi / santhali / kashmiri
    using the plain (_plain.png) variant (consistent with Probe 5b's real
    condition).  Adds N_BLANK blank images (solid white) as the control.

    All images must exist on disk; raises FileNotFoundError otherwise.
    The sample is drawn deterministically from Random(seed) to be
    reproducible without using actual python random state.
    """
    rng = random.Random(seed)
    tasks = []

    for script_name in ("hindi", "santhali", "kashmiri"):
        gt_path = data_root / "raw" / script_name / "ground_truth.jsonl"
        rows = load_ground_truth(gt_path)

        # Keep only plain images that actually exist on disk
        available = [
            r for r in rows
            if (data_root.parent / r["img_plain_path"]).exists()
               or Path(r["img_plain_path"]).exists()
        ]
        if len(available) < N_PER_SCRIPT:
            raise RuntimeError(
                f"Only {len(available)} plain images available for {script_name}; "
                f"need {N_PER_SCRIPT}."
            )

        chosen = rng.sample(available, N_PER_SCRIPT)
        for row in chosen:
            # img_plain_path may be relative to the repo root, not data_root
            img_path = Path(row["img_plain_path"])
            if not img_path.exists():
                img_path = data_root.parent / row["img_plain_path"]
            tasks.append({
                "script": script_name,
                "condition": "real",
                "image_id": row["id"],
                "image_path": str(img_path),
                "ground_truth": row["text"],
                "language_code": LANGUAGE_CODES[script_name],
            })

    # Blank control: solid white PNGs of BLANK_SIZE, one per repetition
    # Written to a temp path under data/cache/sarvam/blanks/ so they are
    # stable across runs (content-addressable cache will de-dup anyway,
    # but having stable paths makes the probe output readable).
    blank_dir = data_root / "cache" / "sarvam" / "blanks"
    blank_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N_BLANK):
        blank_path = blank_dir / f"blank_{i:02d}.png"
        if not blank_path.exists():
            img = Image.new("RGB", BLANK_SIZE, color=(255, 255, 255))
            img.save(blank_path, format="PNG")
        tasks.append({
            "script": "blank",
            "condition": "blank",
            "image_id": f"blank_{i:02d}",
            "image_path": str(blank_path),
            "ground_truth": "",
            "language_code": "hi-IN",  # arbitrary for blank; model sees no text
        })

    return tasks


def load_already_done(out_path: Path) -> set[str]:
    """
    Return the set of (image_path) strings already written to out_path.

    Used for resume: if the probe is interrupted mid-run, re-running it
    will skip already-completed images (whose cache entry will still be
    hit anyway, but skipping keeps the output file clean).
    """
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done.add(rec["image_path"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def run_probe(data_root: Path, out_path: Path) -> None:
    """
    Main entry point.  Submits each task to Sarvam via SarvamClient,
    writes one JSONL record per image as results arrive (incremental),
    and prints a per-image progress line so the user can tell slow from stuck.

    This function does NOT run analysis — that lives in a separate
    analyze_sarvam_transfer.py so the analysis can be re-run offline
    against the cached JSONL without any API calls.
    """
    tasks = build_sample(data_root)
    already_done = load_already_done(out_path)

    pending = [t for t in tasks if t["image_path"] not in already_done]
    total = len(tasks)
    n_skip = total - len(pending)

    logger.info(
        "Budget: %d images total, %d already done, %d to submit to Sarvam API",
        total, n_skip, len(pending),
    )
    logger.info(
        "Estimated cost: ₹%.2f (₹0.5/page × %d new pages)",
        len(pending) * 0.5, len(pending),
    )

    if not pending:
        logger.info("All images already cached — nothing to submit.")
    else:
        logger.info(
            "CONFIRMATION: this will submit %d pages to the Sarvam API, "
            "spending approximately ₹%.2f of your budget.  "
            "Ctrl+C to abort.",
            len(pending), len(pending) * 0.5,
        )

    client = SarvamClient()  # reads SARVAM_API_KEY from env

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Open in append mode so resume works correctly
    with out_path.open("a", encoding="utf-8") as out_f:
        for idx, task in enumerate(tasks, start=1):
            if task["image_path"] in already_done:
                print(
                    f"[{idx:3d}/{total}] SKIP (cached)  {task['script']:10s}  "
                    f"{task['image_id']}",
                    flush=True,
                )
                continue

            print(
                f"[{idx:3d}/{total}] SUBMITTING  {task['script']:10s}  "
                f"{task['image_id']}  → {task['image_path']}",
                flush=True,
            )

            # Override language per-task (kashmiri uses ks-Arab, etc.)
            client.language = task["language_code"]

            try:
                api_result = client.extract_image(task["image_path"])
            except Exception as exc:
                logger.error(
                    "API error for %s / %s: %s", task["script"], task["image_id"], exc
                )
                record = {**task, "error": str(exc), "confidence": None}
            else:
                record = {
                    **task,
                    "cached": api_result["cached"],
                    "job_id": api_result["job_id"],
                    "sarvam_status": api_result["status"],
                    "sarvam_text": api_result.get("result", {}).get("full_text", ""),
                    "confidence": api_result["confidence"],
                    "usage": api_result.get("usage", {}),
                }
                print(
                    f"          confidence={record['confidence']:.4f}  "
                    f"text_preview={record['sarvam_text'][:60]!r}",
                    flush=True,
                )

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    # Print inline summary after all records written
    _print_summary(out_path)


def _print_summary(out_path: Path) -> None:
    """
    Print a concise per-script confidence summary to stdout after the run.

    This is a quick sanity check so the user can immediately see whether
    Sarvam's confidence tracks its own published accuracy gap.
    The full analysis (bootstrap CIs, comparison table, plain-language
    finding) is done offline by analyze_sarvam_transfer.py.
    """
    from collections import defaultdict
    rows: list[dict] = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    by_script: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        c = r.get("confidence")
        if c is not None:
            by_script[r["script"]].append(float(c))

    print("\n=== Sarvam transfer probe — inline summary ===")
    print(f"{'script':<12}  {'n':>3}  {'mean_conf':>10}  {'published_acc':>14}")
    for script in ("hindi", "santhali", "kashmiri", "blank"):
        confs = by_script.get(script, [])
        if not confs:
            continue
        mean_c = sum(confs) / len(confs)
        pub_acc = SARVAM_PUBLISHED_ACCURACY.get(script, float("nan"))
        print(
            f"{script:<12}  {len(confs):>3}  {mean_c:>10.4f}  "
            f"{pub_acc if not isinstance(pub_acc, float) or not (pub_acc != pub_acc) else 'n/a':>14}"
        )

    print(
        "\nKey question: does Sarvam confidence drop in proportion to the "
        "Hindi→Kashmiri accuracy gap (~40pp), or does it stay flat like the "
        "instrument's own confidence did (delta < 0.004)?"
    )
    print(
        "Run analyze_sarvam_transfer.py for bootstrap CIs and the full comparison."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5 Sarvam transfer probe.  "
            "DO NOT run without reading the docstring — this calls a paid API."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Repo data root (default: data/). Raw images and cache live here.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/probe_results/sarvam_transfer_probe.jsonl"),
        help="Output JSONL path (appended to for resume).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the task list and estimated cost without calling the API."
        ),
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        tasks = build_sample(args.data_root)
        already_done = load_already_done(args.out)
        pending = [t for t in tasks if t["image_path"] not in already_done]
        print(f"Total tasks  : {len(tasks)}")
        print(f"Already done : {len(already_done)}")
        print(f"New API calls: {len(pending)}")
        print(f"Estimated ₹  : {len(pending) * 0.5:.2f}")
        print("\nTask list:")
        for t in tasks:
            done_marker = "(done)" if t["image_path"] in already_done else ""
            print(f"  {t['script']:10s}  {t['image_id']:15s}  {t['image_path']}  {done_marker}")
        return

    run_probe(args.data_root, args.out)


if __name__ == "__main__":
    main()

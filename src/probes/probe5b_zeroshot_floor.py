"""
Probe 5b — zero-shot script floor for ungrounded confidence.

Question: if the model reports ~0.99 confidence on real, blank, and
noise Hindi crops (Probe 3), is that just undertraining — would more
steps teach it to condition on the image? This probe removes that
objection structurally: run a Hindi-trained checkpoint on scripts it has
never seen (Santhali Ol Chiki, Kashmiri Perso-Arabic). Zero exposure
by construction — no amount of additional Hindi training could ground
confidence about Ol Chiki.

If mean confidence stays high while the model emits fluent Devanagari
for Ol Chiki images, that is a structural certainty property, not a
training deficiency. Connects to GlotOCR Bench's finding that production
OCR models hallucinate fluent text in a known script rather than
signalling uncertainty on unfamiliar ones.

Called from: Colab after a Hindi Probe 1 checkpoint exists; locally
once checkpoints are present under --output-root.

Compared side-by-side in one run: hindi (in-distribution), santhali,
kashmiri (zero-shot), blank (Probe 3 floor). Accuracy is deliberately
omitted for unseen scripts — see DECISIONS.md #49.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import regex
import torch
from PIL import Image

from probe3_blank_control import make_blank
from probe_utils import load_model_and_tokenizer, resize_to_canonical_height, run_generate

CONDITIONS = ("hindi", "santhali", "kashmiri", "blank")
ZERO_SHOT_LANGUAGES = ("santhali", "kashmiri")
IN_DISTRIBUTION_LANGUAGE = "hindi"

# Unicode blocks for charset composition (aligned with glyph_frequency.py).
SCRIPT_BLOCKS: dict[str, regex.Pattern[str]] = {
    "devanagari": regex.compile(r"[\u0900-\u097F]"),
    "bengali": regex.compile(r"[\u0980-\u09FF]"),
    "ol_chiki": regex.compile(r"[\u1C50-\u1C7F]"),
    "arabic": regex.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"),
}

TRAINING_SCRIPT_BLOCK = {
    "hindi": "devanagari",
    "bengali": "bengali",
}

IMAGE_SCRIPT_BLOCK = {
    "hindi": "devanagari",
    "santhali": "ol_chiki",
    "kashmiri": "arabic",
    "blank": None,
}


def resolve_repo_root(data_root: Path) -> Path:
    """
    Ground-truth jsonl stores repo-relative paths like
    data/raw/santhali/images/0_plain.png. When --data-root is the
    default `data/`, the repo root is its parent.
    """
    return data_root.parent if data_root.name == "data" else data_root


def load_ground_truth_rows(data_root: Path, language: str) -> list[dict]:
    """
    Load Tier C real-image pairs from data/raw/{language}/ground_truth.jsonl.

    These are the GlotOCR-sourced pages already on disk (100 Santhali,
    20 Kashmiri, 60 Hindi) — not synthetic renders.
    """
    gt_path = data_root / "raw" / language / "ground_truth.jsonl"
    if not gt_path.exists():
        raise FileNotFoundError(f"no ground truth at {gt_path}")
    return [json.loads(line) for line in gt_path.read_text(encoding="utf-8").splitlines()]


def resolve_image_path(row: dict, repo_root: Path) -> Path:
    """Resolve img_plain_path from a ground-truth row to an on-disk file."""
    rel = row.get("img_plain_path")
    if not rel:
        raise KeyError(f"ground-truth row missing img_plain_path: {row.get('id')}")
    candidate = Path(rel)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = repo_root / rel
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"image not found: {resolved}")


def grapheme_clusters(text: str) -> list[str]:
    """Segment model output into grapheme clusters for script counting."""
    return regex.findall(r"\X", text)


def classify_grapheme(cluster: str, trained_block: str, image_block: str | None) -> str:
    """
    Bucket one output grapheme for charset composition.

    Returns 'trained', 'image', or 'other'. Whitespace and punctuation
    that match neither script block count as other — the fractions answer
    "what script did the model write in," not "how much whitespace."
    """
    if cluster.strip() == "":
        return "other"
    if SCRIPT_BLOCKS[trained_block].search(cluster):
        return "trained"
    if image_block is not None and SCRIPT_BLOCKS[image_block].search(cluster):
        return "image"
    return "other"


def charset_composition(
    text: str,
    trained_block: str,
    image_block: str | None,
) -> dict:
    """
    Fraction of non-whitespace output graphemes in the trained script
    block vs the script visible in the source image.

    Why graphemes not code points: the instrument decodes grapheme-cluster
    tokens (DECISIONS.md #2). Counting code points would split conjuncts
    and understate Devanagari mass when the model hallucinates fluent Hindi.

    Called from: run_probe5b per record; tested directly in
    tests/test_probe5b_zeroshot_floor.py.
    """
    clusters = grapheme_clusters(text)
    content = [c for c in clusters if c.strip() != ""]
    n = len(content)
    if n == 0:
        return {
            "n_graphemes": 0,
            "n_trained_script": 0,
            "n_image_script": 0,
            "n_other": 0,
            "trained_script_fraction": None,
            "image_script_fraction": None,
            "other_fraction": None,
        }

    n_trained = n_image = n_other = 0
    for cluster in content:
        bucket = classify_grapheme(cluster, trained_block, image_block)
        if bucket == "trained":
            n_trained += 1
        elif bucket == "image":
            n_image += 1
        else:
            n_other += 1

    return {
        "n_graphemes": n,
        "n_trained_script": n_trained,
        "n_image_script": n_image,
        "n_other": n_other,
        "trained_script_fraction": n_trained / n,
        "image_script_fraction": n_image / n,
        "other_fraction": n_other / n,
    }


def load_completed_keys(out_path: Path) -> set[tuple[str, str]]:
    """
    Resume set for incremental jsonl writes.

    Key is (condition, image_path) so a interrupted Colab run picks up
    mid-grid without redoing finished crops.
    """
    if not out_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["condition"], row["image_path"]))
    return done


def build_task_list(
    data_root: Path,
    repo_root: Path,
    n_samples: int,
) -> list[dict]:
    """
    Build the evaluation grid: sampled real images per language plus
    blank controls derived from the same Hindi sample (Probe 3 floor).
    """
    hindi_rows = load_ground_truth_rows(data_root, IN_DISTRIBUTION_LANGUAGE)
    santhali_rows = load_ground_truth_rows(data_root, "santhali")
    kashmiri_rows = load_ground_truth_rows(data_root, "kashmiri")

    rng = random.Random(0)
    hindi_sample = rng.sample(hindi_rows, min(n_samples, len(hindi_rows)))
    santhali_sample = rng.sample(santhali_rows, min(n_samples, len(santhali_rows)))
    kashmiri_sample = rng.sample(kashmiri_rows, min(n_samples, len(kashmiri_rows)))

    tasks: list[dict] = []
    for row in hindi_sample:
        tasks.append({
            "condition": "hindi",
            "language": IN_DISTRIBUTION_LANGUAGE,
            "row": row,
            "image_path": str(resolve_image_path(row, repo_root)),
        })
    for row in santhali_sample:
        tasks.append({
            "condition": "santhali",
            "language": "santhali",
            "row": row,
            "image_path": str(resolve_image_path(row, repo_root)),
        })
    for row in kashmiri_sample:
        tasks.append({
            "condition": "kashmiri",
            "language": "kashmiri",
            "row": row,
            "image_path": str(resolve_image_path(row, repo_root)),
        })
    for row in hindi_sample:
        img_path = str(resolve_image_path(row, repo_root))
        tasks.append({
            "condition": "blank",
            "language": IN_DISTRIBUTION_LANGUAGE,
            "row": row,
            "image_path": img_path,
            "blank_source_path": img_path,
        })
    return tasks


def confidence_histogram(confidences: list[float], n_bins: int = 10) -> list[dict]:
    """Equal-width [0, 1) bins for aggregate confidence distribution."""
    if not confidences:
        return []
    bins = np.linspace(0, 1, n_bins + 1)
    stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = [c for c in confidences if lo <= c < hi]
        if not in_bin:
            continue
        stats.append({
            "range": [round(float(lo), 2), round(float(hi), 2)],
            "n": len(in_bin),
            "mean_confidence": float(np.mean(in_bin)),
        })
    return stats


def summarize_by_condition(records: list[dict]) -> dict:
    """Aggregate mean confidence, charset fractions, and histograms per condition."""
    summary: dict[str, dict] = {}
    for condition in CONDITIONS:
        subset = [r for r in records if r["condition"] == condition]
        if not subset:
            continue
        confs = [r["mean_confidence"] for r in subset if r["mean_confidence"] is not None]
        trained_fracs = [
            r["charset_composition"]["trained_script_fraction"]
            for r in subset
            if r["charset_composition"]["trained_script_fraction"] is not None
        ]
        image_fracs = [
            r["charset_composition"]["image_script_fraction"]
            for r in subset
            if r["charset_composition"]["image_script_fraction"] is not None
        ]
        summary[condition] = {
            "n": len(subset),
            "mean_confidence": float(np.mean(confs)) if confs else None,
            "confidence_std": float(np.std(confs)) if confs else None,
            "confidence_histogram": confidence_histogram(confs),
            "mean_trained_script_fraction": float(np.mean(trained_fracs)) if trained_fracs else None,
            "mean_image_script_fraction": float(np.mean(image_fracs)) if image_fracs else None,
        }
    return summary


def print_summary(summary: dict[str, dict]) -> None:
    """Human-readable table for Colab logs — one block per condition."""
    print("\n[probe5b] === summary by condition ===")
    for condition in CONDITIONS:
        if condition not in summary:
            print(f"  {condition:10s}  (no records)")
            continue
        s = summary[condition]
        conf_str = f"{s['mean_confidence']:.4f}" if s["mean_confidence"] is not None else "n/a"
        trained_str = (
            f"{s['mean_trained_script_fraction']:.3f}"
            if s["mean_trained_script_fraction"] is not None
            else "n/a"
        )
        image_str = (
            f"{s['mean_image_script_fraction']:.3f}"
            if s["mean_image_script_fraction"] is not None
            else "n/a"
        )
        print(
            f"  {condition:10s}  n={s['n']:3d}  "
            f"mean_conf={conf_str}  "
            f"trained_script_frac={trained_str}  "
            f"image_script_frac={image_str}"
        )
        for bucket in s.get("confidence_histogram", []):
            lo, hi = bucket["range"]
            print(f"    conf [{lo:.2f}, {hi:.2f}): n={bucket['n']}  mean={bucket['mean_confidence']:.3f}")


def run_probe5b(
    output_root: Path,
    data_root: Path,
    script: str,
    condition: str,
    seed: int,
    n_samples: int,
    out_path: Path,
    device_str: str = "cpu",
) -> None:
    device = torch.device(device_str)
    model, tokenizer = load_model_and_tokenizer(output_root, script, condition, seed, device)

    repo_root = resolve_repo_root(data_root)
    tasks = build_task_list(data_root, repo_root, n_samples)
    completed = load_completed_keys(out_path)
    pending = [t for t in tasks if (t["condition"], t["image_path"]) not in completed]

    trained_block = TRAINING_SCRIPT_BLOCK[script]
    total = len(tasks)
    already = total - len(pending)

    print(
        f"[probe5b] {script} {condition} seed={seed}: "
        f"{total} tasks ({already} already done, {len(pending)} remaining)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_records.append(json.loads(line))

    with out_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(pending):
            condition_name = task["condition"]
            row = task["row"]
            image_block = IMAGE_SCRIPT_BLOCK[condition_name]

            if condition_name == "blank":
                base = resize_to_canonical_height(Image.open(task["blank_source_path"]))
                image = make_blank(base)
            else:
                image = resize_to_canonical_height(Image.open(task["image_path"]))

            out = run_generate(model, tokenizer, image, device)
            text = out["text"]
            step_confs = out["step_confidences"]
            mean_conf = float(np.mean(step_confs)) if step_confs else None

            record = {
                "condition": condition_name,
                "checkpoint_script": script,
                "training_condition": condition,
                "seed": seed,
                "image_path": task["image_path"],
                "image_id": row.get("id"),
                "ground_truth": row.get("text"),
                "ground_truth_script": row.get("script"),
                "trained_script_block": trained_block,
                "image_script_block": image_block,
                "text": text,
                "mean_confidence": mean_conf,
                "step_confidences": step_confs,
                "charset_composition": charset_composition(text, trained_block, image_block),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            all_records.append(record)

            done_count = already + i + 1
            if done_count % 5 == 0 or done_count == total:
                print(f"[probe5b] {done_count}/{total} done ({condition_name})")

    summary = summarize_by_condition(all_records)
    print_summary(summary)
    print(f"[probe5b] wrote {out_path} ({len(all_records)} records)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe 5b: zero-shot script floor for ungrounded confidence",
    )
    ap.add_argument("--script", required=True, choices=["hindi", "bengali"])
    ap.add_argument("--condition", required=True, choices=["natural", "flattened", "inverted"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", required=True, help="Directory with trained checkpoints")
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OCR_DATA_ROOT", "data"),
        help="Data root containing raw/ ground truth (default: OCR_DATA_ROOT or data/)",
    )
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_probe5b(
        Path(args.output_root),
        Path(args.data_root),
        args.script,
        args.condition,
        args.seed,
        args.n_samples,
        Path(args.out),
        args.device,
    )


if __name__ == "__main__":
    main()

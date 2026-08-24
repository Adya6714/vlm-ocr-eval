"""
Pulls Stage 0's ground-truth image/text pairs from GlotOCR-bench
(cis-lmu/GlotOCR-bench on Hugging Face) and saves them into
data/raw/{hindi,bengali,santhali,kashmiri}/ as image files plus a
ground_truth.jsonl sidecar per folder.

Why this exists: Stage 0 needs (image, ground_truth_text) pairs to
measure OCR engine error rates against. Manually collecting and
transcribing documents doesn't scale and introduces exactly the kind
of ground-truth uncertainty this project is trying to measure *out* of
other people's benchmarks. GlotOCR-bench already has this, rendered
from real multilingual text, with two realism tiers (img_plain,
img_old_document) that map onto this project's Tier A (clean) / Tier B
(degraded) distinction for free.

Called manually, once, before run_baselines.py. Re-run is idempotent —
it overwrites the same output folders.
"""

import json
import os
from datasets import load_dataset

# One entry per target folder: which GlotOCR script config to load,
# and which language codes within that config to keep. GlotOCR groups
# multiple languages under one script config (e.g. Deva holds Hindi,
# Nepali, Marathi, ...), so filtering by language code is required —
# filtering by script alone would silently mix in the wrong language.
TARGETS = {
    "hindi": {
        "config": "Deva",
        "language_codes": {"hin_Deva"},
    },
    "bengali": {
        "config": "Beng",
        "language_codes": {"ben_Beng"},
    },
    "santhali": {
        "config": "Olck",
        "language_codes": {"sat_Olck"},
    },
    "kashmiri": {
        "config": "Arab",
        "language_codes": {"kas_Arab"},
    },
}

OUTPUT_ROOT = "data/raw"


def fetch_and_save(folder_name: str, config: str, language_codes: set[str]) -> None:
    """
    Loads one GlotOCR-bench config, filters to the given language
    code(s), and writes matched rows to data/raw/{folder_name}/.

    Output layout per row:
        data/raw/{folder_name}/images/{id}_plain.png
        data/raw/{folder_name}/images/{id}_degraded.png
        data/raw/{folder_name}/ground_truth.jsonl   (one JSON line per row)

    Kept as separate plain/degraded files rather than a single image
    per row because Stage 0 runs baselines on both realism tiers
    independently (see IMPLEMENTATION.md Stage 1's Tier A/B split) —
    downstream code should not need to know GlotOCR's internal field
    names to find the right image.
    """
    print(f"[{folder_name}] loading config={config} ...")
    ds = load_dataset("cis-lmu/GlotOCR-bench", config)["test"]

    matched = ds.filter(lambda row: row["language"] in language_codes)
    print(f"[{folder_name}] matched {len(matched)} / {len(ds)} rows "
          f"for language codes {language_codes}")

    if len(matched) == 0:
        print(f"[{folder_name}] WARNING: zero rows matched. Check the "
              f"language code — run list_available_languages() below "
              f"for this config to see what's actually present.")
        return

    img_dir = os.path.join(OUTPUT_ROOT, folder_name, "images")
    os.makedirs(img_dir, exist_ok=True)
    gt_path = os.path.join(OUTPUT_ROOT, folder_name, "ground_truth.jsonl")

    with open(gt_path, "w", encoding="utf-8") as gt_file:
        for row in matched:
            plain_path = os.path.join(img_dir, f"{row['id']}_plain.png")
            degraded_path = os.path.join(img_dir, f"{row['id']}_degraded.png")
            row["img_plain"].save(plain_path)
            row["img_old_document"].save(degraded_path)

            gt_file.write(json.dumps({
                "id": row["id"],
                "text": row["text"],
                "language": row["language"],
                "script": row["script"],
                "source": row["source"],
                "img_plain_path": plain_path,
                "img_degraded_path": degraded_path,
            }, ensure_ascii=False) + "\n")

    print(f"[{folder_name}] saved {len(matched)} pairs -> {gt_path}")


def list_available_languages(config: str) -> None:
    """
    Diagnostic helper: prints every distinct language code present in
    a given GlotOCR-bench config. Use this if a TARGETS entry above
    matches zero rows — the assumed language code (e.g. 'kas_Arab')
    may not be exactly what GlotOCR uses internally.
    """
    ds = load_dataset("cis-lmu/GlotOCR-bench", config)["test"]
    codes = sorted(set(ds["language"]))
    print(f"config={config} -> {len(codes)} language codes present:")
    for c in codes:
        print(f"  {c}")


if __name__ == "__main__":
    for folder_name, spec in TARGETS.items():
        fetch_and_save(folder_name, spec["config"], spec["language_codes"])

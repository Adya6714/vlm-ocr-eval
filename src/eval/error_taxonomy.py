"""
Stage 0's actual deliverable: for every prediction any engine made,
classify it as EXACT MATCH, TIER 1 (encoding variant), TIER 2 (phonetic
variant), a genuine-error taxonomy label (from hand_review.py's human-
confirmed notes), or UNREVIEWED (never made it through the hand-review
pass). Then report, per engine, what fraction of all non-exact
predictions falls into each bucket.

Why this exists: this is the report IMPLEMENTATION.md Stage 0 calls
"the input to the interview's strongest single claim" -- what fraction
of reported OCR errors are actually just encoding/phonetic variants
rather than real misreads. Everything before this file (fetch_glotocr,
run_baselines, equivalence_tables, transliteration_equivalence,
hand_review) fed into producing this one number, per engine.

Important design choice: Tier 1 and Tier 2 are RE-COMPUTED here against
current code, not read from hand_review_notes.jsonl's stored
suggestion. Notes were captured at various points while Tier 1/2 were
still being fixed (see DECISIONS.md #18, #23, #24) -- re-checking fresh
means a bug fix to equivalence_tables.py automatically corrects this
report's counts without needing another human review pass. Only rows
that survive Tier 1 AND Tier 2 unexplained fall back to the human label
in hand_review_notes.jsonl; if no human label exists for a row, it's
reported as UNREVIEWED rather than silently dropped or guessed.

Input:
    data/raw/{language}/ground_truth.jsonl
    data/predictions/{engine}/{language}.jsonl
    data/predictions/hand_review_notes.jsonl

Output:
    data/predictions/error_taxonomy.csv   -- one row per non-exact-match prediction
    Printed summary table -- per engine, % EXACT / TIER1 / TIER2 / genuine / unreviewed

Run: python3 src/eval/error_taxonomy.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from equivalence_tables import tier1_equivalent, normalize_tier1
from transliteration_equivalence import tier2_equivalent, SCRIPT_MAP

RAW_ROOT = "data/raw"
PRED_ROOT = "data/predictions"
NOTES_PATH = os.path.join(PRED_ROOT, "hand_review_notes.jsonl")
OUTPUT_CSV = os.path.join(PRED_ROOT, "error_taxonomy.csv")

LANGUAGES = ["hindi", "bengali", "santhali", "kashmiri"]
ENGINES = ["tesseract", "surya", "paddleocr"]

# Labels hand_review.py's allowed-label set can produce. Anything in
# this set counts as a "genuine" error in the summary report; kept as
# an explicit list rather than "anything not Tier 1/2" so a typo'd or
# unexpected label in the notes file surfaces loudly instead of being
# silently counted as genuine.
GENUINE_ERROR_LABELS = {
    "genuine-misread",
    "dropped-matra-nukta",
    "reading-order-break",
    "hallucinated-repeated-text",
}


def load_ground_truth(language: str) -> dict:
    """Indexes ground_truth.jsonl by id. Same shape as the loaders in run_baselines.py / hand_review.py."""
    path = os.path.join(RAW_ROOT, language, "ground_truth.jsonl")
    index = {}
    if not os.path.exists(path):
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["id"]] = row
    return index


def load_predictions(engine: str, language: str) -> list[dict]:
    """Reads one engine/language's raw predictions. Returns [] if the file doesn't exist (engine skipped that language)."""
    path = os.path.join(PRED_ROOT, engine, f"{language}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_human_labels() -> dict:
    """
    Indexes hand_review_notes.jsonl by (language, engine, image_id,
    variant) -> confirmed label. Only rows with a non-null 'label' are
    indexed -- skipped/unresolved rows in the notes file correctly stay
    absent here and fall through to UNREVIEWED in the main report,
    which is more honest than pretending a skip means "not an error."
    """
    index = {}
    if not os.path.exists(NOTES_PATH):
        return index
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("label"):
                key = (row["language"], row.get("engine", "tesseract"),
                       row["image_id"], row["variant"])
                index[key] = row["label"]
    return index


def classify(ground_truth: str, predicted: str, language: str, human_label: str | None) -> tuple[str, str]:
    """
    Returns (bucket, detail) for one prediction, checked in the same
    order every earlier stage of this pipeline has used: exact match,
    then Tier 1, then Tier 2, then fall back to a human-confirmed
    label, then UNREVIEWED if none exists.

    bucket is one of: EXACT, TIER1, TIER2, GENUINE, UNREVIEWED
    detail is the taxonomy label for GENUINE, or empty otherwise.
    """
    if normalize_tier1(ground_truth) == normalize_tier1(predicted):
        # normalize_tier1 already includes whitespace normalization
        # and NFC, so this single check also catches plain exact
        # matches -- no separate raw-string comparison needed.
        if ground_truth.strip() == predicted.strip():
            return "EXACT", ""
        return "TIER1", ""

    if language in SCRIPT_MAP:
        try:
            if tier2_equivalent(ground_truth, predicted, language):
                return "TIER2", ""
        except Exception:
            pass  # malformed/partial predictions can break transliteration;
                   # treat as not-Tier-2-explained rather than crashing

    if human_label:
        if human_label not in GENUINE_ERROR_LABELS:
            print(f"WARNING: unexpected label '{human_label}' not in "
                  f"GENUINE_ERROR_LABELS -- check hand_review_notes.jsonl "
                  f"for a typo. Counting it as genuine anyway.")
        return "GENUINE", human_label

    return "UNREVIEWED", ""


def main() -> None:
    human_labels = load_human_labels()
    ground_truth_by_language = {lang: load_ground_truth(lang) for lang in LANGUAGES}

    # per-engine counts for the summary table
    counts = {engine: {"EXACT": 0, "TIER1": 0, "TIER2": 0, "GENUINE": 0, "UNREVIEWED": 0}
              for engine in ENGINES}
    taxonomy_breakdown = {engine: {} for engine in ENGINES}  # engine -> {label: count}

    csv_rows = []

    for engine in ENGINES:
        for language in LANGUAGES:
            predictions = load_predictions(engine, language)
            gt_index = ground_truth_by_language[language]

            for pred_row in predictions:
                if pred_row.get("skipped_reason"):
                    continue  # engine didn't attempt this language/image -- not a scored prediction

                gt_row = gt_index.get(pred_row["id"])
                if gt_row is None:
                    continue

                predicted_text = pred_row.get("predicted_text") or ""
                ground_truth_text = gt_row["text"]

                human_label = human_labels.get(
                    (language, engine, pred_row["id"], pred_row["variant"])
                )

                bucket, detail = classify(ground_truth_text, predicted_text, language, human_label)
                counts[engine][bucket] += 1
                if bucket == "GENUINE":
                    taxonomy_breakdown[engine][detail] = taxonomy_breakdown[engine].get(detail, 0) + 1

                if bucket != "EXACT":
                    csv_rows.append({
                        "engine": engine,
                        "language": language,
                        "image_id": pred_row["id"],
                        "variant": pred_row["variant"],
                        "bucket": bucket,
                        "taxonomy_label": detail,
                        "ground_truth": ground_truth_text,
                        "predicted": predicted_text,
                    })

    # --- write the CSV: one row per non-exact-match prediction ---
    os.makedirs(PRED_ROOT, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "engine", "language", "image_id", "variant", "bucket",
            "taxonomy_label", "ground_truth", "predicted",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {len(csv_rows)} non-exact-match rows -> {OUTPUT_CSV}\n")

    # --- print the summary report: this is Stage 0's actual deliverable ---
    print("=" * 78)
    print("STAGE 0 SUMMARY -- fraction of all predictions per bucket, per engine")
    print("(EXACT is a correct read outright; TIER1/TIER2 are 'not actually")
    print(" errors'; GENUINE is a real error; UNREVIEWED has no human label yet)")
    print("=" * 78)

    for engine in ENGINES:
        total = sum(counts[engine].values())
        if total == 0:
            print(f"\n[{engine}] no predictions found (skipped or run_baselines.py "
                  f"not yet run for this engine)")
            continue

        print(f"\n[{engine}]  (n={total})")
        for bucket in ["EXACT", "TIER1", "TIER2", "GENUINE", "UNREVIEWED"]:
            n = counts[engine][bucket]
            pct = 100 * n / total
            print(f"  {bucket:12s} {n:5d}  ({pct:5.1f}%)")

        if taxonomy_breakdown[engine]:
            print(f"  --- GENUINE breakdown ---")
            for label, n in sorted(taxonomy_breakdown[engine].items(), key=lambda x: -x[1]):
                print(f"    {label:28s} {n}")

        # The headline number: of everything that LOOKED like an error
        # (i.e. not EXACT), what fraction turned out to be Tier 1 or
        # Tier 2 rather than a genuine misread. This is the "published
        # Indic OCR numbers are quietly pessimistic" claim, computed
        # directly rather than asserted.
        non_exact = total - counts[engine]["EXACT"]
        if non_exact > 0:
            not_really_errors = counts[engine]["TIER1"] + counts[engine]["TIER2"]
            pct_not_real = 100 * not_really_errors / non_exact
            print(f"  --- headline ---")
            print(f"  Of {non_exact} predictions that weren't an exact match, "
                  f"{not_really_errors} ({pct_not_real:.1f}%) were Tier 1/Tier 2 "
                  f"variants, not real errors.")

    if any(counts[e]["UNREVIEWED"] > 0 for e in ENGINES):
        print("\nNOTE: UNREVIEWED rows exist -- these never went through "
              "hand_review.py, or the notes file doesn't cover them yet. "
              "They're excluded from the GENUINE breakdown above but "
              "counted in the totals. Run hand_review.py further to "
              "close this gap, or treat the current report as provisional.")


if __name__ == "__main__":
    main()

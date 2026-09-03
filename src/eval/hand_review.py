"""
Side-by-side viewer for Stage 0's hand-reading pass.

Why this exists: IMPLEMENTATION.md Stage 0 requires reading real engine
output by hand BEFORE writing error_taxonomy.py, so the taxonomy
categories come from what you actually observe, not from guessing.
This script makes that hour of reading fast: it shows ground truth next
to each engine's prediction, and pre-applies Tier 0 whitespace
normalization plus Tier 1 + Tier 2 so you're not re-reading things
that already aren't errors.

Input: data/raw/{language}/ground_truth.jsonl (from fetch_glotocr.py)
       data/predictions/{engine}/{language}.jsonl (from run_baselines.py)

What it does: for a given language, walks through each image, prints
ground truth vs. every engine's prediction, flags whether Tier 0/1/2
already explain the difference, and — for UNEXPLAINED cases — shows a
suggested residual label from hand_review_assist.py before asking for
a keypress. Labels accumulate in a running notes file you can turn into
error_taxonomy.py's category list afterward.

For UNEXPLAINED cases, hand_review_assist.py proposes one of four
fixed residual labels (or explicitly refuses). The proposal is shown
first; Enter confirms it, a typed label overrides, `s` skips. The
agent never writes a label without that keypress.

Run: python3 src/eval/hand_review.py hindi
     python3 src/eval/hand_review.py bengali --engine surya
     PYTHONPATH=src/eval python3 src/eval/hand_review.py --self-test
     PYTHONPATH=src/eval python3 src/eval/hand_review.py \\
         --queue data/predictions/adjudication_sample.jsonl
"""

import argparse
import json
import os

from equivalence_tables import normalize_whitespace, tier1_equivalent
from hand_review_assist import SUGGESTED_LABELS, Suggestion, suggest_unexplained_label
from transliteration_equivalence import tier2_equivalent, SCRIPT_MAP

RAW_ROOT = "data/raw"
PRED_ROOT = "data/predictions"
NOTES_PATH = "data/predictions/hand_review_notes.jsonl"


def load_ground_truth(language: str) -> dict:
    """
    Indexes ground_truth.jsonl by id for fast lookup while walking
    through predictions. Same loader shape as run_baselines.py's, kept
    separate here rather than imported to avoid this review tool
    depending on run_baselines.py's engine-running code at all.
    """
    path = os.path.join(RAW_ROOT, language, "ground_truth.jsonl")
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["id"]] = row
    return index


def load_predictions(engine: str, language: str) -> list[dict]:
    """Reads one engine's predictions for one language, in the order run_baselines.py wrote them."""
    path = os.path.join(PRED_ROOT, engine, f"{language}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def classify_against_tiers(ground_truth: str, predicted: str, language: str) -> str:
    """
    Runs a (ground_truth, predicted) pair through Tier 0 whitespace
    normalization, then Tier 1, then Tier 2 — same order
    error_taxonomy.py will use later — and returns a short label for
    what's already explained, so the human reviewer's attention goes
    to what ISN'T explained yet, which is the point of this pass.
    """
    if normalize_whitespace(ground_truth) == normalize_whitespace(predicted):
        return "EXACT MATCH (after whitespace normalization)"
    if tier1_equivalent(ground_truth, predicted):
        return "TIER 1 (encoding variant, not an error)"
    if language in SCRIPT_MAP:
        try:
            if tier2_equivalent(ground_truth, predicted, language):
                return "TIER 2 (phonetic variant, not an error)"
        except Exception:
            pass  # Tier 2 can fail on partial/garbled predictions; that
                   # failure itself is informative, fall through to
                   # "UNEXPLAINED" rather than crashing the review pass
    return "UNEXPLAINED -- read this one"


def _note_from_response(
    language: str,
    image_id: str,
    variant: str,
    engine: str,
    ground_truth: str,
    predicted: str,
    suggestion: Suggestion,
    typed: str,
) -> dict:
    """
    Turns one reviewer keypress into a notes-file row.

    Why this is separate from the print/input loop: the outcome
    (agent-suggested-and-confirmed vs human-overridden vs skipped)
    has to be assigned the same way every time, including the
    no-fit case where Enter cannot confirm a missing label. Keeping
    that mapping in one function means the jsonl schema stays
    stable even if the prompt wording changes.

    Enter with a real suggestion confirms it. Enter with no
    suggestion, or `s`/`skip`, is a skip — still written, so we
    can see which proposals were ignored. `q` is handled by the
    caller and never reaches here.
    """
    skip_tokens = {"s", "skip"}
    is_skip = typed.lower() in skip_tokens or (typed == "" and not suggestion.fits)
    if is_skip:
        outcome = "skipped"
        label = None
    elif typed == "":
        outcome = "agent-suggested-and-confirmed"
        label = suggestion.label
    elif suggestion.fits and typed == suggestion.label:
        outcome = "agent-suggested-and-confirmed"
        label = typed
    else:
        outcome = "human-overridden"
        label = typed

    return {
        "language": language,
        "image_id": image_id,
        "variant": variant,
        "engine": engine,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "label": label,
        "suggested_label": suggestion.label,
        "suggested_reason": suggestion.reason,
        "suggestion_fits": suggestion.fits,
        "suggestion_outcome": outcome,
    }


def _note_key(note: dict) -> tuple[str, str, str, str]:
    return (
        note.get("engine", "tesseract"),
        note["language"],
        note["image_id"],
        note["variant"],
    )


def load_labeled_keys(notes_path: str = NOTES_PATH) -> set[tuple[str, str, str, str]]:
    """Keys that already have a non-null label in the notes file."""
    keys: set[tuple[str, str, str, str]] = set()
    if not os.path.isfile(notes_path):
        return keys
    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            note = json.loads(line)
            if note.get("label") and note.get("suggestion_outcome") not in {
                "pending-adjudication",
                "skipped",
            }:
                keys.add(_note_key(note))
    return keys


def review_queue(queue_path: str, notes_path: str = NOTES_PATH) -> None:
    """
    Interactive adjudication over a pre-drawn sample queue.

    Why this exists: adjudication_sample.py writes UNREVIEWED rows in
    the same notes schema this module already uses. Rather than a
    second review UI, --queue walks that file, skips keys already
    labeled in NOTES_PATH, and appends completed notes there so
    error_taxonomy.py / adjudication bootstrap pick them up.

    Allowed residual labels are the Stage 0 four. Reviewers who find a
    Tier 1 table gap may type encoding-variant (or tier1 / exact-match)
    — those strings are counted toward the Tier 1 numerator in
    src/analysis/adjudication_sample.py bootstrap.
    """
    with open(queue_path, "r", encoding="utf-8") as f:
        queue = [json.loads(line) for line in f if line.strip()]

    already = load_labeled_keys(notes_path)
    pending = [row for row in queue if _note_key(row) not in already]
    print(
        f"Queue {queue_path}: {len(queue)} rows, "
        f"{len(pending)} unlabeled "
        f"({len(queue) - len(pending)} already in {notes_path})"
    )
    if not pending:
        print("Nothing left to adjudicate.")
        return

    os.makedirs(os.path.dirname(notes_path) or ".", exist_ok=True)
    notes_file = open(notes_path, "a", encoding="utf-8")

    encoding_hint = (
        "encoding-variant / tier1 / exact-match = not a real error "
        "(Tier 1 table gap)"
    )
    for i, row in enumerate(pending):
        engine = row.get("engine", "tesseract")
        language = row["language"]
        image_id = row["image_id"]
        variant = row["variant"]
        ground_truth = row["ground_truth"]
        predicted = row["predicted"]

        print("\n" + "=" * 70)
        print(
            f"[{i + 1}/{len(pending)}] engine={engine} language={language} "
            f"id={image_id} variant={variant}"
        )
        print(f"GROUND TRUTH: {ground_truth}")
        print(f"PREDICTED:    {predicted!r}")
        classification = classify_against_tiers(ground_truth, predicted, language)
        print(f"auto tiers:   {classification}")

        suggestion = suggest_unexplained_label(ground_truth, predicted)
        print(f"  --- review [{engine}] ---")
        if suggestion.fits:
            print(f"  SUGGESTED:    {suggestion.label}")
        else:
            print("  SUGGESTED:    (none of the four labels fit well)")
        print(f"  REASON:       {suggestion.reason}")
        print(f"  Allowed residual labels: {', '.join(SUGGESTED_LABELS)}")
        print(f"  Or type: {encoding_hint}")
        raw = input(
            "  Enter=confirm suggestion, type a label to override, "
            "'s' to skip, 'q' to quit: "
        )
        typed = raw.strip()
        if typed.lower() == "q":
            break
        note = _note_from_response(
            language=language,
            image_id=image_id,
            variant=variant,
            engine=engine,
            ground_truth=ground_truth,
            predicted=predicted,
            suggestion=suggestion,
            typed=typed,
        )
        # Preserve sample provenance when present.
        if row.get("adjudication_sample"):
            note["adjudication_sample"] = True
            note["sample_seed"] = row.get("sample_seed")
            note["sample_index"] = row.get("sample_index")
        notes_file.write(json.dumps(note, ensure_ascii=False) + "\n")
        notes_file.flush()
        print(
            f"  recorded [{note['suggestion_outcome']}] "
            f"label={note['label']!r}"
        )

    notes_file.close()
    print(f"\nSession notes appended to {notes_path}")


def review_session(language: str, engines: list[str]) -> None:
    """
    Main interactive loop. Walks every image for the given language,
    shows ground truth + each requested engine's prediction + its Tier
    classification. For each UNEXPLAINED engine output, shows a
    suggested residual label from hand_review_assist.py (or an explicit
    no-fit), then waits for Enter (confirm), a typed label (override),
    `s` (skip), or `q` (quit). Notes accumulate in NOTES_PATH across
    runs (append mode) so this can be stopped and resumed freely —
    the point is building intuition and a note trail, not finishing
    in one sitting.

    Each note records suggestion_outcome so later analysis can tell
    agent-suggested-and-confirmed from human-overridden, instead of
    treating every label as if a human invented it.
    """
    ground_truth_index = load_ground_truth(language)
    predictions_by_engine = {e: load_predictions(e, language) for e in engines}

    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    notes_file = open(NOTES_PATH, "a", encoding="utf-8")

    # Iterate in the order run_baselines.py wrote predictions -- use
    # the first engine's list as the walk order, all engines were run
    # over the same image set so this is safe.
    for pred_row in predictions_by_engine[engines[0]]:
        image_id = pred_row["id"]
        variant = pred_row["variant"]
        gt_row = ground_truth_index.get(image_id)
        if gt_row is None:
            continue  # shouldn't happen, but don't crash the session over it

        print("\n" + "=" * 70)
        print(f"image id={image_id} variant={variant}  (source: {gt_row.get('source')})")
        print(f"IMAGE FILE: {pred_row['image_path']}")
        print(f"GROUND TRUTH: {gt_row['text']}")

        unexplained: list[tuple[str, str]] = []
        for engine in engines:
            matching = [p for p in predictions_by_engine[engine]
                        if p["id"] == image_id and p["variant"] == variant]
            if not matching:
                continue
            pred = matching[0]
            predicted_text = pred.get("predicted_text") or ""
            skipped = pred.get("skipped_reason")

            if skipped:
                print(f"  [{engine}] SKIPPED: {skipped}")
                continue

            classification = classify_against_tiers(gt_row["text"], predicted_text, language)
            print(f"  [{engine}] PREDICTED: {predicted_text!r}")
            print(f"  [{engine}] -> {classification}")
            if "UNEXPLAINED" in classification:
                unexplained.append((engine, predicted_text))

        quit_session = False
        for engine, predicted_text in unexplained:
            suggestion = suggest_unexplained_label(gt_row["text"], predicted_text)
            print(f"  --- review [{engine}] ---")
            print(f"  GROUND TRUTH: {gt_row['text']}")
            print(f"  PREDICTED:    {predicted_text!r}")
            if suggestion.fits:
                print(f"  SUGGESTED:    {suggestion.label}")
            else:
                print("  SUGGESTED:    (none of the four labels fit well)")
            print(f"  REASON:       {suggestion.reason}")
            print(f"  Allowed labels: {', '.join(SUGGESTED_LABELS)}")
            raw = input(
                "  Enter=confirm suggestion, type a label to override, "
                "'s' to skip, 'q' to quit: "
            )
            typed = raw.strip()
            if typed.lower() == "q":
                quit_session = True
                break
            note = _note_from_response(
                language=language,
                image_id=image_id,
                variant=variant,
                engine=engine,
                ground_truth=gt_row["text"],
                predicted=predicted_text,
                suggestion=suggestion,
                typed=typed,
            )
            notes_file.write(json.dumps(note, ensure_ascii=False) + "\n")
            notes_file.flush()
            print(
                f"  recorded [{note['suggestion_outcome']}] "
                f"label={note['label']!r}"
            )
        if quit_session:
            break

    notes_file.close()
    print(f"\nSession notes appended to {NOTES_PATH}")


def run_note_outcome_checks() -> int:
    """
    Checkable mapping from keypress -> suggestion_outcome.

    Why it lives here: the outcome strings are a contract with later
    analysis of the notes file, not with the string-comparison
    heuristic. A regression that silently turned Enter into skip
    would poison that analysis.
    """
    suggested = Suggestion(
        label="genuine-misread",
        reason="test",
        fits=True,
    )
    no_fit = Suggestion(label=None, reason="no fit", fits=False)
    cases = [
        ("", suggested, "agent-suggested-and-confirmed", "genuine-misread"),
        ("genuine-misread", suggested, "agent-suggested-and-confirmed", "genuine-misread"),
        ("dropped-matra-nukta", suggested, "human-overridden", "dropped-matra-nukta"),
        ("s", suggested, "skipped", None),
        ("skip", suggested, "skipped", None),
        ("", no_fit, "skipped", None),
        ("genuine-misread", no_fit, "human-overridden", "genuine-misread"),
    ]
    failures = 0
    for typed, suggestion, expected_outcome, expected_label in cases:
        note = _note_from_response(
            language="hindi",
            image_id="x",
            variant="plain",
            engine="tesseract",
            ground_truth="a",
            predicted="b",
            suggestion=suggestion,
            typed=typed,
        )
        ok = (
            note["suggestion_outcome"] == expected_outcome
            and note["label"] == expected_label
        )
        if not ok:
            failures += 1
        status = "OK" if ok else "FAIL"
        print(
            f"[{status}] typed={typed!r} fits={suggestion.fits} -> "
            f"outcome={note['suggestion_outcome']!r} label={note['label']!r} "
            f"(expected {expected_outcome!r}, {expected_label!r})"
        )
    print(f"\n{len(cases) - failures}/{len(cases)} note-outcome checks passed")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0 hand-reading viewer")
    parser.add_argument(
        "language",
        nargs="?",
        choices=["hindi", "bengali", "santhali", "kashmiri"],
        help="Required unless --self-test or --queue.",
    )
    parser.add_argument("--engine", action="append", dest="engines",
                         help="Engine to include (repeatable). Default: all three.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run note-outcome checks (no interactive session).",
    )
    parser.add_argument(
        "--queue",
        metavar="PATH",
        help=(
            "Adjudicate a pre-drawn sample jsonl (notes schema) instead of "
            "walking a language. Writes completed labels to "
            f"{NOTES_PATH}."
        ),
    )
    parser.add_argument(
        "--notes",
        default=NOTES_PATH,
        help=f"Notes jsonl to append (default {NOTES_PATH}).",
    )
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_note_outcome_checks())
    if args.queue:
        review_queue(args.queue, notes_path=args.notes)
        raise SystemExit(0)
    if not args.language:
        parser.error("language is required unless --self-test or --queue is set")

    engines = args.engines or ["tesseract", "surya", "paddleocr"]
    review_session(args.language, engines)

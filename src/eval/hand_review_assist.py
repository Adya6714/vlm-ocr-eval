"""
Suggested taxonomy labels for Stage 0's UNEXPLAINED hand-review cases.

Why a separate file: hand_review.py is the interactive viewer and
should stay that. The suggestion logic is a guess with a checkable
right answer on constructed pairs, so it needs its own tests and must
not be mixed into the prompt loop. The human still decides; this module
only proposes. See DECISIONS.md #20.

Sits between classify_against_tiers (hand_review.py) and the typed
label that gets written to hand_review_notes.jsonl. It never talks to
an engine or the Sarvam API. The four labels are the residual buckets
from IMPLEMENTATION.md Stage 0 after Tier 1/2 have already been ruled
out — encoding/phonetic variants must not be suggested here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import unicodedata

from equivalence_tables import normalize_whitespace

# Fixed set. Do not add categories here because a pair is awkward —
# return label=None instead. Names match IMPLEMENTATION.md Stage 0's
# residual buckets (hyphenated for the notes file).
SUGGESTED_LABELS = (
    "genuine-misread",
    "dropped-matra-nukta",
    "reading-order-break",
    "hallucinated-repeated-text",
)

# How many consecutive copies of the same token (or of the whole GT
# string) counts as "the decoder got stuck," not a one-off duplication.
# Two copies can be a layout echo; three is the usual OCR loop.
_REPEAT_RUN_MIN = 3

# Below this many whitespace tokens, "same bag, different order" is
# too easy to confuse with a two-word substitution.
_MIN_TOKENS_FOR_ORDER = 3

# Relative Levenshtein above this, on similar-length strings, is
# treated as unrelated dumps rather than a letter-level misread.
_GENUINE_MISREAD_MAX_RATIO = 0.75


@dataclass(frozen=True)
class Suggestion:
    """
    One proposal for an UNEXPLAINED (gt, pred) pair.

    label is one of SUGGESTED_LABELS, or None when none of the four
    fit well. fits is False exactly when label is None — kept as a
    separate flag so callers don't treat a missing label as
    "skip silently." reason is always filled, including for no-fit,
    so the reviewer can see why the agent refused to force a bucket.
    """

    label: str | None
    reason: str
    fits: bool


def _prepare_for_heuristic(ground_truth: str, predicted: str) -> tuple[str, str]:
    """
    Tier 0 pass shared with classify_against_tiers() in hand_review.py.

    Every heuristic in this module must diff only these normalized
    strings — never ground_truth/predicted raw. Internal newlines and
    extra spaces from line-wrapping are layout noise, not letter edits;
    passing raw strings into _levenshtein inflated genuine-misread
    (DECISIONS.md #22).
    """
    return normalize_whitespace(ground_truth), normalize_whitespace(predicted)


def suggest_unexplained_label(ground_truth: str, predicted: str) -> Suggestion:
    """
    Best-guess residual taxonomy bucket for one UNEXPLAINED pair.

    Why this exists: the hand-review hour is supposed to produce the
    category list for error_taxonomy.py, but a blank prompt invites
    inconsistent free text ("matra", "missing vowel", "nukta drop").
    A constrained suggestion keeps labels in the four Stage 0 residual
    buckets while still requiring a keypress (DECISIONS.md #20).

    Called from: hand_review.py, once per UNEXPLAINED engine output,
    after Tier 1/2 have already said the pair is not an encoding or
    phonetic variant. Hands off a Suggestion; never writes the notes
    file itself.

    Applies normalize_whitespace() first — same Tier 0 pass as
    classify_against_tiers() in hand_review.py, before any string diff.
    Without it, internal newlines or extra spaces from line-wrapping
    inflate genuine-misread instead of being treated as layout noise.

    Priority (first match that fires) is documented in DECISIONS.md
    #20: repetition, then permutation, then matra/nukta-only, then
    letter-level substitution, else explicit no-fit.

    Raw inputs are normalized once via _prepare_for_heuristic(); all
    signals below operate on those strings only.
    """
    gt, pred = _prepare_for_heuristic(ground_truth, predicted)

    if not gt and not pred:
        return Suggestion(
            label=None,
            reason="both strings empty after whitespace normalization; none of the four buckets apply",
            fits=False,
        )
    if gt == pred:
        return Suggestion(
            label=None,
            reason=(
                "strings match after whitespace normalization; "
                "this should have been EXACT MATCH, not UNEXPLAINED"
            ),
            fits=False,
        )

    repeated = _repeated_text_signal(gt, pred)
    if repeated is not None:
        return Suggestion(
            label="hallucinated-repeated-text",
            reason=repeated,
            fits=True,
        )

    order = _reading_order_signal(gt, pred)
    if order is not None:
        return Suggestion(
            label="reading-order-break",
            reason=order,
            fits=True,
        )

    matra = _dropped_matra_nukta_signal(gt, pred)
    if matra is not None:
        return Suggestion(
            label="dropped-matra-nukta",
            reason=matra,
            fits=True,
        )

    misread = _genuine_misread_signal(gt, pred)
    if misread is not None:
        return Suggestion(
            label="genuine-misread",
            reason=misread,
            fits=True,
        )

    return Suggestion(
        label=None,
        reason=_no_fit_reason(gt, pred),
        fits=False,
    )


def _is_matra_or_nukta(ch: str) -> bool:
    """
    True for Indic vowel signs (matras) and nuktas only.

    Virama/halant is kept: dropping it turns a conjunct into two
    consonants, which is a different error than a missing matra.
    Anusvara/visarga/chandrabindu are also kept — they are not
    matras or nuktas, so a pair that differs only there must fall
    through to no-fit rather than be forced into dropped-matra-nukta.

    Unicode names, not code-point ranges, so Devanagari and Bengali
    (and any other Indic block with standard names) share one check.
    """
    name = unicodedata.name(ch, "")
    if "NUKTA" in name:
        return True
    if "VIRAMA" in name:
        return False
    return "VOWEL SIGN" in name


def strip_matras_and_nuktas(text: str) -> str:
    """
    Removes matras and nuktas after NFD, leaving base letters.

    NFD first because a precomposed akshara (NFC) hides the matra as
    part of one code point; stripping without decomposing would miss
    the entire class of errors this signal is for.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not _is_matra_or_nukta(ch))


def _levenshtein(a: str, b: str) -> int:
    """
    Classic edit distance; used only as a similarity signal, not as
    the reported metric. Callers must pass Tier-0-normalized strings.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row DP: previous row is distance against a[:i], current against a[:i+1].
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _tokens(text: str) -> list[str]:
    return text.split()


def _max_consecutive_run(tokens: list[str]) -> tuple[int, str | None]:
    """Longest run of the same token in a row, and which token it was."""
    if not tokens:
        return 0, None
    best_run = 1
    best_tok = tokens[0]
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run > best_run:
                best_run = run
                best_tok = tokens[i]
        else:
            run = 1
    return best_run, best_tok


def _phrase_repeat_run(tokens: list[str], phrase_len: int) -> int:
    """
    How many times a phrase of `phrase_len` tokens tiles the tail of
    the sequence. OCR loops often repeat a 2–4 word chunk, not just
    a single token.
    """
    if phrase_len < 1 or len(tokens) < phrase_len * _REPEAT_RUN_MIN:
        return 0
    phrase = tokens[-phrase_len:]
    run = 1
    i = len(tokens) - 2 * phrase_len
    while i >= 0 and tokens[i : i + phrase_len] == phrase:
        run += 1
        i -= phrase_len
    return run


def _repeated_text_signal(gt: str, pred: str) -> str | None:
    """
    Detect decoder loops / copied spans, the hallucination this
    bucket is named for — not merely 'pred is longer than gt'.

    gt and pred must already be Tier-0-normalized (single spaces).
    """
    pred_tokens = _tokens(pred)
    gt_tokens = _tokens(gt)

    run, tok = _max_consecutive_run(pred_tokens)
    gt_count = Counter(gt_tokens).get(tok, 0) if tok else 0
    if run >= _REPEAT_RUN_MIN and tok and run > gt_count + 1:
        return (
            f"predicted token {tok!r} repeats consecutively {run} times "
            f"(ground truth has it {gt_count} time(s))"
        )

    for n in (2, 3, 4):
        phrase_run = _phrase_repeat_run(pred_tokens, n)
        if phrase_run >= _REPEAT_RUN_MIN:
            phrase = " ".join(pred_tokens[-n:])
            return (
                f"predicted {n}-word span {phrase!r} tiles the output "
                f"{phrase_run} times"
            )

    gt_compact = " ".join(gt_tokens)
    pred_compact = " ".join(pred_tokens)
    if gt_compact and pred_compact.count(gt_compact) >= 2:
        copies = pred_compact.count(gt_compact)
        return (
            f"ground-truth string appears {copies} times inside the prediction"
        )

    return None


def _reading_order_signal(gt: str, pred: str) -> str | None:
    """
    Same words, different sequence. Requires a true permutation of
    the token multiset so a one-word substitution cannot sneak in.
    """
    gt_tokens = _tokens(gt)
    pred_tokens = _tokens(pred)
    if len(gt_tokens) < _MIN_TOKENS_FOR_ORDER or len(pred_tokens) < _MIN_TOKENS_FOR_ORDER:
        return None
    if gt_tokens == pred_tokens:
        return None
    if Counter(gt_tokens) != Counter(pred_tokens):
        return None
    return (
        f"same {len(gt_tokens)} whitespace tokens in a different order "
        "(multiset matches, sequence does not)"
    )


def _dropped_matra_nukta_signal(gt: str, pred: str) -> str | None:
    """
    Bases match once matras and nuktas are stripped, so the residual
    difference is only those marks — the error this bucket names.
    """
    gt_base = strip_matras_and_nuktas(gt)
    pred_base = strip_matras_and_nuktas(pred)
    if gt_base != pred_base:
        return None
    if gt_base == gt and pred_base == pred:
        # Nothing was stripped, so the visible difference is not
        # matra/nukta (whitespace-only would already have been an
        # exact-match return in suggest_unexplained_label).
        return None
    return (
        "base letters match after NFD-stripping vowel signs and nuktas; "
        "the remaining difference is those marks"
    )


def _genuine_misread_signal(gt: str, pred: str) -> str | None:
    """
    Letter-level substitution of similar-length strings, after the
    more specific buckets have already declined.
    """
    if not gt or not pred:
        return None
    max_len = max(len(gt), len(pred))
    min_len = min(len(gt), len(pred))
    # Extreme length mismatch without a repetition signal is a dump or
    # a total omission, not a grapheme confusion.
    if min_len / max_len < 0.4:
        return None
    dist = _levenshtein(gt, pred)
    ratio = dist / max_len
    if ratio > _GENUINE_MISREAD_MAX_RATIO:
        return None
    if dist == 0:
        return None
    return (
        f"base-level substitution (edit distance {dist}/{max_len} chars), "
        "not a permutation, loop, or matra/nukta-only diff"
    )


def _no_fit_reason(gt: str, pred: str) -> str:
    """Explain the refusal so the reviewer is not staring at a blank 'none'."""
    if not pred:
        return "prediction is empty; omission of the whole span is not one of the four labels"
    if not gt:
        return "ground truth is empty but prediction is not; none of the four labels cover that"
    max_len = max(len(gt), len(pred))
    min_len = min(len(gt), len(pred))
    if min_len / max_len < 0.4:
        return (
            "length mismatch without a repetition loop — extra/missing span, "
            "not a clean instance of any of the four labels"
        )
    dist = _levenshtein(gt, pred)
    return (
        f"none of the four labels fit well (edit distance {dist}/{max_len}; "
        "not a token permutation, not matra/nukta-only, not a repeat loop)"
    )


# Constructed pairs with a known bucket. These are the checkable
# property this module has — run via `python3 src/eval/hand_review_assist.py`.
# Real engine output is allowed to disagree; the human keypress is the
# label of record.
VALIDATION_PAIRS: list[tuple[str, str, str | None]] = [
    ("राम", "राम राम राम राम", "hallucinated-repeated-text"),
    (
        "एक दो तीन चार",
        "एक दो तीन चार एक दो तीन चार एक दो तीन चार",
        "hallucinated-repeated-text",
    ),
    ("राम सीता लक्ष्मण", "सीता राम लक्ष्मण", "reading-order-break"),
    ("किताब", "कताब", "dropped-matra-nukta"),  # ि dropped
    ("फ़न", "फन", "dropped-matra-nukta"),  # nukta dropped
    ("কাজ", "কজ", "dropped-matra-nukta"),  # Bengali া dropped
    ("राम", "श्याम", "genuine-misread"),
    ("कमल", "कपाळ", "genuine-misread"),
    ("राम", "", None),
    ("", "something invented", None),
    # Unrelated scripts, similar length, no loop: not a letter confusion
    # inside the same writing system, and none of the other three fire.
    ("किताब", "hello!", None),
    # Tier 0 whitespace only — must not land in any residual bucket.
    ("राम सीता", "राम\n\nसीता", None),
    # id=153-shaped: internal newline mid-sentence, letters identical.
    (
        "यूनानी भाषा में इक्वूलियस एक छोटे घोड़े को या घोड़े के बच्चे को कहते हैं।",
        "यूनानी भाषा में इक्वूलियस एक छोटे घोड़े को\nया घोड़े के बच्चे को कहते हैं।",
        None,
    ),
]


def run_validation() -> int:
    """
    Asserts each VALIDATION_PAIRS entry lands in the expected bucket
    (or no-fit). Returns the number of failures so the process exit
    code can fail a CI-less local run.
    """
    failures = 0
    for gt, pred, expected in VALIDATION_PAIRS:
        got = suggest_unexplained_label(gt, pred)
        ok = got.label == expected
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"[{status}] suggest({gt!r}, {pred!r}) -> {got.label!r} "
            f"(expected {expected!r}) reason={got.reason!r}"
        )
    print(f"\n{len(VALIDATION_PAIRS) - failures}/{len(VALIDATION_PAIRS)} validation pairs passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(run_validation())

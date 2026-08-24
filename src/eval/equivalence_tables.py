"""
Tier 1: deterministic encoding equivalence.

These are cases where two different Unicode byte sequences render as,
or are read as, the exact same thing — not a judgment call, an
uncontroversial fact about the writing system. See DECISIONS.md #4 and
#8 for why this is kept separate from Tier 2 (transliteration_equivalence.py),
which handles judgment-call phonetic equivalence instead.

Important honesty note: the table below is a STARTING scaffold, seeded
from known Unicode phenomena in Devanagari and Bengali, not a finished
artifact. IMPLEMENTATION.md Stage 0 says this table gets built FROM the
hand-reading pass over real engine output — once run_baselines.py
finishes and you've read ~30 predictions by hand, come back and add
whatever variant pairs you actually observed that aren't covered yet.
Treat every entry here as "known to exist in the literature," not
"confirmed to show up in your engines' output."

NFC normalization (composed vs. decomposed Unicode forms) is NOT
duplicated here on purpose — olmOCR-bench, which Sarvam's own eval
wraps, already NFC-normalizes before scoring (DECISIONS.md #4). Adding
it again here would just be redundant, not wrong, but it's left out to
keep this file's job narrowly scoped to what NFC does NOT already cover.
"""

import re
import unicodedata

# Each pair: (variant_form, canonical_form). normalize_tier1 replaces
# every occurrence of variant_form with canonical_form. Order matters
# only if one pattern is a substring of another — longer patterns are
# applied first below, see _sort_by_length.
TIER1_PAIRS = [
    # --- Danda vs. Latin punctuation ---
    # Devanagari/Bengali sentence-final punctuation vs. Latin period.
    # A model outputting either is making the same reading, not a
    # different one.
    ("\u0964", "."),   # danda ।
    ("\u0965", ".."),  # double danda ॥
    # Pipe character as a danda substitute -- some OCR engines render
    # the danda's vertical stroke as a plain ASCII pipe rather than the
    # Unicode danda codepoint or a period. Observed directly in Stage 0
    # hand-review (image ids 222-225, DECISIONS.md #23) as a false
    # genuine-misread: same sentence, engine just chose "|" over "।".
    ("|", "."),
    ("||", ".."),

    # --- Digit systems ---
    # Devanagari digits -> Latin. A model that reads a Devanagari-script
    # page but emits Latin digits (common, since digits are often
    # trained/tokenized separately from script) should not be penalized
    # for this alone.
    ("\u0966", "0"), ("\u0967", "1"), ("\u0968", "2"), ("\u0969", "3"),
    ("\u096A", "4"), ("\u096B", "5"), ("\u096C", "6"), ("\u096D", "7"),
    ("\u096E", "8"), ("\u096F", "9"),

    # Bengali digits -> Latin, same reasoning.
    ("\u09E6", "0"), ("\u09E7", "1"), ("\u09E8", "2"), ("\u09E9", "3"),
    ("\u09EA", "4"), ("\u09EB", "5"), ("\u09EC", "6"), ("\u09ED", "7"),
    ("\u09EE", "8"), ("\u09EF", "9"),

    # --- Khanda-ta (Bengali) ---
    # ৎ (U+09CE, khanda-ta, the "broken ta" used word-finally) is
    # visually and phonetically identical to ত + ্ + ZWJ
    # (ta + virama + zero-width joiner) in the contexts where khanda-ta
    # is used. A model choosing one encoding over the other is not
    # making a reading error.
    ("\u09A4\u09CD\u200D", "\u09CE"),  # ত্‍ (ta+virama+ZWJ) -> ৎ

    # --- ZWJ/ZWNJ around virama (both scripts) ---
    # Zero-width joiner/non-joiner control whether a consonant cluster
    # renders as a visually fused conjunct or a half-form + explicit
    # virama mark. Both are valid encodings of the same underlying
    # reading; stripping ZWJ/ZWNJ around a virama normalizes this away.
    ("\u094D\u200D", "\u094D"),  # Devanagari virama+ZWJ -> bare virama
    ("\u094D\u200C", "\u094D"),  # Devanagari virama+ZWNJ -> bare virama
    ("\u09CD\u200D", "\u09CD"),  # Bengali virama+ZWJ -> bare virama
    ("\u09CD\u200C", "\u09CD"),  # Bengali virama+ZWNJ -> bare virama

    # --- Nukta-formed letters: precomposed vs. base+nukta ---
    # These specific pairs are already covered by NFC, kept here ONLY
    # as a defensive no-op in case a caller runs Tier 1 on text that
    # was never NFC-normalized (e.g. testing this file in isolation).
    ("\u0915\u093C", "\u0958"),  # क + nukta -> क़ (qa)
    ("\u0916\u093C", "\u0959"),  # ख + nukta -> ख़ (kha, Persian-origin)
    ("\u0917\u093C", "\u095A"),  # ग + nukta -> ग़ (ghain)
    ("\u091C\u093C", "\u095B"),  # ज + nukta -> ज़ (za)
    ("\u0921\u093C", "\u095C"),  # ड + nukta -> ड़ (dda)
    ("\u0922\u093C", "\u095D"),  # ढ + nukta -> ढ़ (ddha)
    ("\u092B\u093C", "\u095E"),  # फ + nukta -> फ़ (fa)

    # Bengali nukta-formed letters, same reasoning.
    ("\u09A1\u09BC", "\u09DC"),  # ড + nukta -> ড় (rra)
    ("\u09A2\u09BC", "\u09DD"),  # ঢ + nukta -> ঢ় (rha)
    ("\u09AF\u09BC", "\u09DF"),  # য + nukta -> য় (yya)

    # TODO (fill in after Stage 0's hand-reading pass): whatever
    # actual variant pairs show up in run_baselines.py output that
    # aren't covered above. This list is a floor, not a ceiling.
]

# --- Anusvara sandhi (Devanagari) ---
# Anusvara (ं) before a class consonant is a well-defined, deterministic
# spelling convention for that consonant's homorganic nasal -- e.g.
# हिन्दी (न् explicit) and हिंदी (ं) are the same word, same
# pronunciation, just two accepted spellings under classical Sanskrit
# sandhi rules. This is NOT a Tier 2 case: it isn't a judgment call
# about whether two spellings sound similar, it's a fixed rule with one
# right answer, so it belongs here in Tier 1 rather than being left to
# transliteration_equivalence.py. (See DECISIONS.md #18 for how this
# was originally misclassified as Tier 2 and why that was wrong --
# aksharamukha's ISO 15919 output transliterates anusvara literally as
# 'ṁ' rather than resolving it, so Tier 2 alone cannot catch this.)
_DEVANAGARI_NASAL_CLASSES = {
    "\u0915\u0916\u0917\u0918\u0919": "\u0919",  # velar क ख ग घ -> ङ
    "\u091A\u091B\u091C\u091D\u091E": "\u091E",  # palatal च छ ज झ -> ञ
    "\u091F\u0920\u0921\u0922\u0923": "\u0923",  # retroflex ट ठ ड ढ -> ण
    "\u0924\u0925\u0926\u0927\u0928": "\u0928",  # dental त थ द ध -> न
    "\u092A\u092B\u092C\u092D\u092E": "\u092E",  # labial प फ ब भ -> म
}
_VIRAMA = "\u094D"
_ANUSVARA = "\u0902"


def resolve_anusvara_sandhi(text: str) -> str:
    """
    Canonicalizes explicit homorganic-nasal spellings down to the
    anusvara form (chosen as canonical arbitrarily -- either direction
    works, this just needs to be consistent). E.g. सम्बन्ध -> संबंध.
    Called from normalize_tier1 as an extra pass; kept as a separate
    function because it needs class-lookup logic, not a flat
    string-replace pair like the rest of TIER1_PAIRS.
    """
    for consonants, nasal in _DEVANAGARI_NASAL_CLASSES.items():
        for consonant in consonants:
            text = text.replace(nasal + _VIRAMA + consonant, _ANUSVARA + consonant)
    return text


def _sort_by_length(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Ensures longer patterns are checked before their substrings, so a
    3-codepoint pattern like virama+ZWJ gets matched before a shorter
    pattern that might otherwise fire on part of it first.
    """
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


_SORTED_TIER1_PAIRS = _sort_by_length(TIER1_PAIRS)


def normalize_whitespace(text: str) -> str:
    """
    Collapses all whitespace -- spaces, tabs, and newlines -- to a
    single space, and strips the ends. This is deliberately NOT folded
    into normalize_tier1(): whitespace/line-break differences are a
    LAYOUT artifact (an engine wrapped a line differently, or a
    multi-column page got read in a different line order), not an
    encoding or phonetic equivalence. Reading order itself is measured
    properly in Stage 3 (Kendall tau) -- this function's only job is
    to stop line-wrapping noise from being miscounted as a text error
    in Stage 0 and Probe 4, where the question is "did it read the
    right characters," not "did it lay them out the same way."

    Called first, before either equivalence tier, in both
    hand_review.py's classify_against_tiers and (once built)
    error_taxonomy.py's alignment step.
    """
    return " ".join(text.split())


def normalize_tier1(text: str) -> str:
    """
    Applies every Tier 1 encoding-equivalence substitution, plus an
    NFC pass first (belt-and-suspenders alongside olmOCR-bench's own
    NFC normalization, since this function may be called standalone
    outside that pipeline).

    Two strings are Tier-1-equivalent if normalize_tier1(a) ==
    normalize_tier1(b). Called from error_taxonomy.py once
    run_baselines.py has produced predictions to compare.
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_whitespace(text)
    text = resolve_anusvara_sandhi(text)
    for variant, canonical in _SORTED_TIER1_PAIRS:
        text = text.replace(variant, canonical)
    # Strip whitespace immediately before terminal punctuation, applied
    # AFTER the pair substitutions above so danda/pipe variants have
    # already been canonicalized to '.' by this point -- one regex
    # then covers "है ।" / "है ." / "है |" as the same case. Observed
    # directly in Stage 0 hand-review as a second false genuine-misread
    # source alongside the pipe-vs-danda substitution (DECISIONS.md #23).
    text = re.sub(r"\s+(\.+)", r"\1", text)
    return text


def tier1_equivalent(a: str, b: str) -> bool:
    """Convenience wrapper: are these two strings the same after Tier 1 normalization?"""
    return normalize_tier1(a) == normalize_tier1(b)


if __name__ == "__main__":
    # Minimal smoke test — not the real validation set (that's Tier 2's
    # 40-pair set per IMPLEMENTATION.md), just confirms the mechanics
    # work before this file is imported elsewhere.
    smoke_tests = [
        ("२०२४", "2024", True),           # Devanagari digits vs Latin
        ("রাত্‍রি", "রাত্‍রি", True),        # identical strings, trivially equal
        ("तिम्रा।", "तिम्रा.", True),        # danda vs period
        ("अ", "आ", False),                 # genuinely different letters
        ("हिन्दी", "हिंदी", True),          # anusvara sandhi: nasal consonant vs anusvara
        ("सम्बन्ध", "सम्बंध", True),        # same rule, different word
        ("है ।", "है।", True),              # space-before-danda, id=222
        ("यह।", "यह|", True),               # pipe as danda substitute
        ("रात॥", "रात ..", True),           # double danda vs spaced double period
    ]
    for a, b, expected in smoke_tests:
        actual = tier1_equivalent(a, b)
        status = "OK" if actual == expected else "FAIL"
        print(f"[{status}] tier1_equivalent({a!r}, {b!r}) = {actual}, expected {expected}")

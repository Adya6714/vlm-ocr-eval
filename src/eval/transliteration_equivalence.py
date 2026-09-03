"""
Tier 2: phonetic/transliteration equivalence.

Unlike Tier 1 (equivalence_tables.py), this is a judgment call, not an
uncontroversial encoding fact: are two differently-spelled strings the
"same word"? Handled by transliterating both into a shared canonical
Latin form (ISO 15919) and comparing there, rather than hand-enumerating
every phonetic-equivalence case. See DECISIONS.md #8 for why this
mechanism was chosen over a rule table or an LLM-primary scorer.

Reported SEPARATELY from Tier 1 in error_taxonomy.py's output — this
tier can be wrong or arguable in ways Tier 1 can't, so collapsing them
into one number would hide that distinction.

Scope honesty (DECISIONS.md #18, #54):
- Pairs must be *truly phonetically identical*, not conventional
  spelling variants (राम/रामः, कमल/कमाल fail correctly — different sounds).
- Anusvara ↔ homorganic nasal (हिन्दी/हिंदी) is phonetic identity but
  is Tier 1 sandhi, not Tier 2: aksharamukha emits ṁ literally, so
  tier2_equivalent returns False. Those pairs are boundary tests here,
  not positives.
- Many positives below (nukta NFC/NFD, virama+ZWJ) also normalize under
  Tier 1. That is fine for validating this function in isolation;
  error_taxonomy.py runs Tier 1 first, so corpus-level TIER2 counts only
  the residual cases ISO collapses that Tier 1 did not (e.g. ॐ/ओं,
  Bengali ৎ vs bare ত্ without ZWJ).
"""

from __future__ import annotations

from aksharamukha import transliterate

# Maps this project's internal language folder names to the aksharamukha
# source-script identifier. aksharamukha only supports Brahmic-family
# scripts for reliable ISO 15919 transliteration -- Santhali (Ol Chiki)
# and Kashmiri (Perso-Arabic) are deliberately absent, per DECISIONS.md
# #8's stated boundary: Tier 2 does not extend to non-Brahmic scripts.
SCRIPT_MAP = {
    "hindi": "Devanagari",
    "bengali": "Bengali",
}


def to_canonical(text: str, language: str) -> str:
    """
    Transliterates one string into ISO 15919 Latin form for the given
    language's script. Raises KeyError for languages outside Tier 2's
    scope (santhali, kashmiri) -- callers should check SCRIPT_MAP
    membership before calling, not catch this as a normal case.
    """
    script = SCRIPT_MAP[language]
    return transliterate.process(script, "ISO", text)


def tier2_equivalent(ref: str, hyp: str, language: str) -> bool:
    """
    Are these two strings the same word once transliterated to a
    shared canonical form? True doesn't mean "identical characters" --
    it means "the same underlying sound sequence," which is the
    looser, arguable claim Tier 2 is explicitly scoped to make (as
    opposed to Tier 1's uncontroversial encoding-equivalence claim).
    """
    return to_canonical(ref, language) == to_canonical(hyp, language)


# ---------------------------------------------------------------------------
# Validation set (DECISIONS.md #18 / #54, IMPLEMENTATION.md Stage 0)
#
# Each entry: (reference, hypothesis, language, expected_equivalent)
#
# Built from classes where ISO 15919 *actually* collapses two Unicode
# spellings of the same sound sequence. Conventional spelling variants
# that differ in sound are negatives or omitted (see #18). Unverified
# candidates stay as TODO comments, not invented positives.
#
# Ratio: ~2 positives : 1 genuine-different negative, plus a few
# explicit scope-boundary rows (phonetic identity that Tier 2 correctly
# does *not* claim — anusvara sandhi belongs in Tier 1).
# ---------------------------------------------------------------------------

# Combining nukta (U+093C Devanagari / U+09BC Bengali) — used so the
# decomposed form is visibly distinct from the precomposed letter in
# source even when editors NFC-normalize on save.
_DN = "\u093C"  # Devanagari nukta
_BN = "\u09BC"  # Bengali nukta
_DZ = "\u200D"  # ZWJ
_DNJ = "\u200C"  # ZWNJ
_DV = "\u094D"  # Devanagari virama
_BV = "\u09CD"  # Bengali virama

VALIDATION_SET: list[tuple[str, str, str, bool]] = [
    # ----- Positives: nukta precomposed ↔ base+nukta (same letter) -----
    # These are also Tier 1 / NFC; included so the transliteration path
    # is checked on real Persian-loan letters that OCR often flips.
    ("क़िताब", "क" + _DN + "िताब", "hindi", True),   # qa
    ("फ़ोन", "फ" + _DN + "ोन", "hindi", True),       # fa
    ("ज़मीन", "ज" + _DN + "मीन", "hindi", True),     # za
    ("ख़ुशी", "ख" + _DN + "ुशी", "hindi", True),     # k͟ha
    ("ग़लत", "ग" + _DN + "लत", "hindi", True),       # ġa
    ("बड़ा", "बड" + _DN + "ा", "hindi", True),       # ṛa
    ("पढ़ा", "पढ" + _DN + "ा", "hindi", True),       # ṛha
    ("ড়", "ড" + _BN, "bengali", True),              # rra
    ("ঢ়", "ঢ" + _BN, "bengali", True),              # rha
    ("য়", "য" + _BN, "bengali", True),              # yya
    ("হয়", "হয" + _BN, "bengali", True),            # word with yya

    # ----- Positives: ZWJ / ZWNJ in conjuncts (same reading) -----
    # Tier 1 strips virama+ZWJ/ZWNJ; ISO also ignores the joiner.
    ("क्ष", "क" + _DV + _DZ + "ष", "hindi", True),
    ("अक्षर", "अक" + _DV + _DZ + "षर", "hindi", True),
    ("प्रज्ञा", "प्रज" + _DV + _DZ + "ञा", "hindi", True),
    ("लक्ष्मण", "लक" + _DV + _DZ + "ष्मण", "hindi", True),
    ("धर्म", "धर्" + _DZ + "म", "hindi", True),  # ZWJ after repha virama
    ("ক্ষমা", "ক" + _BV + _DNJ + "ষমা", "bengali", True),
    ("বিদ্যালয়", "বিদ" + _BV + _DZ + "যালয়", "bengali", True),

    # ----- Positives: Tier-2-only residuals (Tier 1 does NOT equate) -----
    # ॐ and ओं share ISO ōṁ; Tier 1 leaves both unchanged.
    ("ॐ", "ओं", "hindi", True),
    ("ॐ नमः", "ओं नमः", "hindi", True),
    # Bengali khanda-ta vs bare ta+virama (Tier 1 only maps ত্‍+ZWJ → ৎ).
    ("ৎ", "ত" + _BV, "bengali", True),
    ("উৎসব", "উত" + _BV + "সব", "bengali", True),
    # ZWNJ not adjacent to virama — Tier 1's virama+joiner rule misses it;
    # ISO still drops the control char, same sound sequence.
    ("भारत", "भा" + _DNJ + "रत", "hindi", True),

    # ----- Negatives: genuinely different words / sounds -----
    ("राम", "श्याम", "hindi", False),
    ("कमल", "कमला", "hindi", False),       # lotus vs name (extra ā)
    ("कमल", "कमाल", "hindi", False),       # short a vs long ā (#18)
    ("दिन", "दीन", "hindi", False),         # i vs ī
    ("सुर", "सूर", "hindi", False),         # u vs ū
    ("बाल", "बल", "hindi", False),           # ā vs a
    ("नमः", "नमस्", "hindi", False),       # visarga ≠ sa+virama in ISO
    ("आম", "আমি", "bengali", False),
    ("ভালো", "ভাল", "bengali", False),     # trailing o vs bare
    ("দিন", "দীন", "bengali", False),       # i vs ī
    ("নদী", "নদি", "bengali", False),       # ī vs i
    ("শ", "স", "bengali", False),           # śa vs sa — different phonemes

    # ----- Scope boundary: phonetic identity Tier 2 must NOT claim -----
    # Anusvara ↔ nasal is true phonetic identity under sandhi, but
    # aksharamukha keeps ṁ; Tier 1 resolve_anusvara_sandhi owns these.
    ("हिन्दी", "हिंदी", "hindi", False),
    ("गङ्गा", "गंगा", "hindi", False),
    ("পণ্ডিত", "পংডিত", "bengali", False),  # if ISO keeps ṁ / ṇ distinct
]

# TODO — do NOT invent; verify before promoting to VALIDATION_SET:
# - Devanagari eyelash-repha / Marathi ऱ contexts that OCR emits as र
#   but speakers treat as the same flap in a specific word (dialect-
#   dependent; need a cited pair, not a guess).
# - Bengali অ vs অ্যা loanword pairs where both spellings are accepted
#   for the same English source — often different ISO vowels; confirm
#   against a dictionary before treating as identity.
# - Kashmiri/Santhali: out of Tier 2 scope (SCRIPT_MAP); no pairs.


def run_validation() -> None:
    """
    Runs VALIDATION_SET through tier2_equivalent and reports pass/fail
    per pair, plus an overall pass rate. This is the check
    IMPLEMENTATION.md Stage 0 requires before trusting Tier 2 on the
    full corpus -- run this, look at every failure by hand, and decide
    whether the failure is a bad validation-set entry or a real gap in
    aksharamukha's transliteration before moving on.
    """
    passed = 0
    scored = 0
    for ref, hyp, language, expected in VALIDATION_SET:
        if language not in SCRIPT_MAP:
            print(f"[SKIP] {language} not in Tier 2 scope: {ref!r} vs {hyp!r}")
            continue
        scored += 1
        try:
            actual = tier2_equivalent(ref, hyp, language)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {ref!r} vs {hyp!r} ({language}): {e}")
            continue
        status = "OK" if actual == expected else "FAIL"
        if actual == expected:
            passed += 1
        print(
            f"[{status}] tier2_equivalent({ref!r}, {hyp!r}, {language!r}) "
            f"= {actual}, expected {expected}"
        )

    print(f"\n{passed}/{scored} validation pairs passed")
    n_pos = sum(1 for *_, exp in VALIDATION_SET if exp)
    n_neg = sum(1 for *_, exp in VALIDATION_SET if not exp)
    print(f"(set composition: {n_pos} expected-True, {n_neg} expected-False)")


if __name__ == "__main__":
    run_validation()

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
"""

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


# Starter validation set. IMPLEMENTATION.md Stage 0 calls for ~40
# hand-picked pairs before this tier is trusted on the full corpus --
# these ~10 are a seed, not the finished set. Add more once Stage 0's
# hand-reading pass over real engine output surfaces real disagreement
# cases, especially ones this function gets wrong (those are the ones
# worth adding).
#
# Each entry: (reference, hypothesis, language, expected_equivalent)
VALIDATION_SET = [
    # Genuinely equivalent: same word, different but both-valid spelling
    ("श्री", "shri", "hindi", False),  # NOTE: mixed-script pairs aren't
        # what this function compares -- both inputs must be in the
        # SAME script for a meaningful transliteration comparison. Left
        # in deliberately as an example of a MISUSE case this function
        # should not be asked to judge; see the assertion note below.

    # NOTE: anusvara-vs-homorganic-nasal pairs (e.g. हिन्दी / हिंदी)
    # were tried here and moved OUT -- they turned out to be a Tier 1
    # case, not Tier 2. aksharamukha's ISO 15919 transliterates anusvara
    # literally as 'ṁ' rather than resolving it, so this function
    # cannot catch that equivalence -- but the rule for when anusvara
    # equals which nasal is fully deterministic (classical sandhi), so
    # it belongs in equivalence_tables.py's resolve_anusvara_sandhi
    # instead, where it now lives. See DECISIONS.md #18.
    #
    # TODO: find a genuine Tier 2 Devanagari pair -- something that's
    # a real judgment call about phonetic similarity, not a fixed rule
    # -- once Stage 0's hand-reading pass surfaces a real candidate.

    # Genuinely different words -- must NOT be flagged equivalent
    ("राम", "श्याम", "hindi", False),
    ("कमल", "कमला", "hindi", False),     # different word (lotus vs a name), not a spelling variant

    # TODO: Bengali true-phonetic-identity pairs. Left empty rather
    # than filled with unverified guesses -- the same anusvara/nasal
    # alternation likely exists in Bengali, but needs a confirmed real
    # example before going in the validation set, not an invented one.
    # Fill this in once Stage 0's hand-reading pass surfaces a real
    # candidate, or from a source you can actually verify pronunciation
    # against.

    # Bengali genuinely different
    ("আম", "আমি", "bengali", False),
]


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
    for ref, hyp, language, expected in VALIDATION_SET:
        if language not in SCRIPT_MAP:
            print(f"[SKIP] {language} not in Tier 2 scope: {ref!r} vs {hyp!r}")
            continue
        try:
            actual = tier2_equivalent(ref, hyp, language)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {ref!r} vs {hyp!r} ({language}): {e}")
            continue
        status = "OK" if actual == expected else "FAIL"
        if actual == expected:
            passed += 1
        print(f"[{status}] tier2_equivalent({ref!r}, {hyp!r}, {language!r}) "
              f"= {actual}, expected {expected}")

    total = sum(1 for _, _, lang, _ in VALIDATION_SET if lang in SCRIPT_MAP)
    print(f"\n{passed}/{total} validation pairs passed")


if __name__ == "__main__":
    run_validation()

# Tier 2 validation set — Stage 0

**Date:** 2026-09-03  
**Code:** `src/eval/transliteration_equivalence.py`  
**Decisions:** #8 (ISO 15919 mechanism), #18 (strict phonetic identity),
#54 (this set + corpus finding)

## What was wrong

The in-file `VALIDATION_SET` was a stub: a few negative pairs and TODOs.
With no trusted positives, a corpus-level **TIER2 = 0%** readout was
indistinguishable from “Tier 2 not implemented.”

## What the set is now

`PYTHONPATH=src/eval python3 src/eval/transliteration_equivalence.py`
→ **38/38 passed** (23 expected-True, 15 expected-False).

| Class | Role | Examples |
|-------|------|----------|
| Nukta precomposed ↔ base+nukta | Positive (also Tier 1 / NFC) | क़िताब, फ़ोन, হয় |
| Virama + ZWJ/ZWNJ in conjuncts | Positive (also Tier 1 strip) | क्ष / क्‍ष, ক্ষমা |
| ॐ ↔ ओं | Positive, **Tier-2-only** residual | ISO both `ōṁ` |
| Bengali ৎ ↔ ত্ (no ZWJ) | Positive, **Tier-2-only** | উৎসব / উত্সব |
| Mid-word ZWNJ off virama | Positive, **Tier-2-only** | भारत / भा‌रत |
| Genuinely different words / lengths | Negative | राम≠श्याम, কমল≠কমাল |
| Anusvara ↔ nasal (sandhi) | Boundary expected-False | हिन्दी/हिंदी — Tier 1 owns these |

Unverified dialect / loanword candidates remain **TODO comments** in
the source — not invented positives (honest ~38 beats fabricated 40).

## Corpus re-run (`error_taxonomy.py`)

After the set landed, Stage 0 was re-scored on the existing engine
jsonl:

| Engine | n | EXACT | TIER1 | TIER2 | GENUINE | UNREVIEWED |
|--------|---|-------|-------|-------|---------|------------|
| tesseract | 180 | 15.6% | 17.2% | **0.0%** | 7.2% | 60.0% |
| surya | 222 | 46.8% | 9.0% | **0.0%** | 0.0% | 44.1% |
| paddleocr | 10 | 0.0% | 10.0% | **0.0%** | 0.0% | 90.0% |

**Finding:** Tier 2 stays at **0%** even with a working validation set.
That is not a broken tier. `classify()` runs Tier 1 before Tier 2; the
encoding classes that dominate this corpus (danda/pipe, digits, joiner
stripping, nukta NFC, anusvara sandhi) are already absorbed as TIER1.
Residuals that are phonetically identical *and* still differ after
Tier 1 (ॐ/ओं-class, bare-ৎ, stray ZWNJ) simply do not appear in the
current prediction/GT diffs. Phonetic-variant errors are rare here
relative to encoding-variant errors — state that explicitly; do not
read 0% TIER2 as “not implemented.”

## How to regenerate

```bash
PYTHONPATH=src/eval python3 src/eval/transliteration_equivalence.py
PYTHONPATH=src/eval python3 src/eval/error_taxonomy.py
PYTHONPATH=src python3 -m pytest tests/test_transliteration_equivalence.py -q
```

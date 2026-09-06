# Surya positive control — plumbing (Step 5)

**Status:** not run. No instrument-style per-step logits were collected
from Surya in this session.

## Why Surya

`docs/paper_defensibility_stats.md` Follow-Up 7: Surya exact-match
**46.8%** (104/222) on this repo’s Stage 0 prediction rows. That is a
measured competence on the same GlotOCR-derived pages, unlike Singh
2026. The 60-image Hindi eval set is a subset of that pool (plain
pages), not the 222-row engine table.

## How this repo currently calls Surya

`src/eval/run_baselines.py` `_get_surya_recognizer` / `_surya_page_text`:

- Package: `surya-ocr` **0.22** API (`surya.inference.SuryaInferenceManager`,
  `surya.recognition.RecognitionPredictor`).
- Input: full page PNG (`img_plain` / `img_old_document`).
- Output used: concatenated HTML-stripped block text. **No per-step
  logits, no encoder memory, no token ids.**

That is enough for Stage 0 string metrics. It is **not** enough for
Table 6-style teacher-forced log p(GT) by position.

## Plumbing approach (before any run)

1. **Do not** wrap `RecognitionPredictor` page HTML. Open Surya’s
   recognition decoder (typically a transformer decoder over encoder
   features) and call the same greedy / teacher-forced loop the
   instrument uses: encoder forward → decoder step 0 softmax → force
   next grapheme or Surya subword.
2. **Vocabulary alignment:** Surya does not emit this repo’s 367
   Devanagari grapheme clusters. Plan: decode Surya ids to Unicode,
   then segment with `regex.findall(r"\X", ...)` (same CER segmenter as
   `docs/training_config.md`). Teacher forcing then maps each GT
   grapheme to the **shortest Surya token prefix** that NFC-matches
   that cluster, or records a skip if no prefix exists. Mismatch rate
   must be reported; it is part of the result.
3. **Blank:** white image of the same resized 70 px height as
   `probe_utils.resize_to_canonical_height` so the control matches the
   instrument protocol, plus a second run at native page size if the
   first is OOD for Surya’s detector.
4. **Encoder zeroing:** only if Surya exposes encoder memory as a
   tensor before cross-attn. If the public API hides it, stop and
   report unrecoverable rather than hacking a compiled CUDA kernel.
5. **Agree/flip KL:** same prefix rule as the instrument ablation
   (`docs/training_config.md` Step 0a): zero-encoder re-score under
   **full-memory greedy tokens**, not GT, if that path exists.

Until checkpoints/API access for that decoder loop are in the checkout,
items (1)–(5) of the user spec are **not computed**.

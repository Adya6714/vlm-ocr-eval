# Stage 5b — rank-correlation transfer

Pre-registered in `DECISIONS.md` **#89**. Do not swap Spearman
for Kendall/Pearson after seeing a coefficient.

## Protocol (locked before looking at ρ)

- **Statistic:** Spearman rho between instrument difficulty
  (negative mean over seeds of teacher-forced mean_log_p_gt on condition=real (higher = the instrument finds the page harder)) and production error (Tier-1 grapheme-cluster CER of the production engine vs GT).
- **Sign:** both axes increase with hardness, so a useful ranking
  transfer is **positive** ρ.
- **Null:** shuffle production CER, 10000 permutations, RNG seed
  0, two-sided p = (1 + #{|ρ*| ≥ |ρ_obs|}) / (1 + N).
- **Primary:** Hindi, same `image_id`, Tier A (`*_plain.png`), Sarvam.
- **Secondary (Decision #15):** same Hindi ids on Tier B degraded,
  only if those Extract pages exist in cache/jsonl.
- **Secondary (no extra API):** Tesseract and Surya CER on the same
  Hindi plains. PaddleOCR is listed the same way; it is not a
  matched instrument (Decision #86).
- **Not primary:** pooling Hindi+Santhali+Kashmiri. Language is a
  confounder of image difficulty. Stratified n=10 arms from Stage 5a
  may be reported separately and are underpowered.
- **Not this probe:** per-glyph-class error rates in the original
  IMPLEMENTATION.md 5b bullet. This session's unit is per-image
  (explicit override; see #88).

## Spend

Stage 5a already spent ₹17.50 (35 pages). New Extract calls are
gated by `src/probes/stage5b_sarvam_pages.py --i-confirm-spend-inr`.
At ₹0.5/page: 50 remaining Hindi plains = **₹25.00**; 60 Hindi
degraded = **₹30.00**; both = **₹55.00** (110 pages).

## Results

**Not computed.** Spend has not been confirmed, and this
report must not contain a coefficient until the user confirms
the page set (including the ₹0 / n=10 cache-only option).

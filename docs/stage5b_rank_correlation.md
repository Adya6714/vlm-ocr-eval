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

### Primary: Hindi plains, Sarvam CER

- Statistic: Spearman rho (Decision #89)
- ρ = **0.0293**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.8267**
- n = **60**

Ids: 20, 21, 22, 23, 24, 47, 48, 49, 50, 51, 137, 138, 139, 140, 141, 152, 153, 154, 155, 156, 177, 178, 179, 180, 181, 222, 223, 224, 225, 226, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 292, 293, 294, 295, 296, 307, 308, 309, 310, 311, 317, 318, 319, 320, 321, 356, 357, 358, 359, 360

### Secondary: Hindi degraded, Sarvam CER

- Statistic: Spearman rho (Decision #89)
- ρ = **-0.2222**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.2913**
- n = **24**

### Secondary (no API): Hindi plains, tesseract CER

- Statistic: Spearman rho (Decision #89)
- ρ = **-0.0808**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.5453**
- n = **60**

### Secondary (no API): Hindi plains, surya CER

- Statistic: Spearman rho (Decision #89)
- ρ = **0.2348**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.0685**
- n = **60**

### Secondary (no API): Hindi plains, paddleocr CER

- Statistic: Spearman rho (Decision #89)
- ρ = **-0.1593**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.2173**
- n = **60**

### Exploratory, underpowered: santhali plains (instrument = −Probe5b mean_confidence, not GT log p)

- Statistic: Spearman rho (Decision #89)
- ρ = **-0.3891**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.2632**
- n = **10**

### Exploratory, underpowered: kashmiri plains (instrument = −Probe5b mean_confidence, not GT log p)

- Statistic: Spearman rho (Decision #89)
- ρ = **-0.5394**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.1182**
- n = **10**

Local engines on the full 60-id Hindi pool (independent of Sarvam n):

### Full Hindi-60 pool, tesseract CER (no extra API)

- Statistic: Spearman rho (Decision #89)
- ρ = **-0.0808**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.5453**
- n = **60**

### Full Hindi-60 pool, surya CER (no extra API)

- Statistic: Spearman rho (Decision #89)
- ρ = **0.2348**
- permutation p (two-sided, 10000 shuffles, seed 0) = **0.0685**
- n = **60**

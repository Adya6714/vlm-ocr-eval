# Adjudication sample + Tier 1 rate + ranking

**Generated:** 2026-09-03  
**Code:** `src/analysis/adjudication_sample.py`  
**Labels:** not fabricated — sample queue for humans; bootstrap CI appears only after `hand_review.py --queue`.

---

## 1. Sample

- UNREVIEWED population: **215**
- Sampled: **200** (requested 200, seed=42)
- Taxonomy non-exact denominator: **280**
- Output: `data/predictions/adjudication_sample.jsonl`

| Stratum (engine/language) | n sampled |
|---------------------------|----------:|
| paddleocr/hindi | 8 |
| surya/bengali | 18 |
| surya/hindi | 35 |
| surya/santhali | 39 |
| tesseract/bengali | 39 |
| tesseract/hindi | 61 |

Label with:
```bash
PYTHONPATH=src/eval python3 src/eval/hand_review.py --queue data/predictions/adjudication_sample.jsonl
```

Typed overrides recognized as encoding-variant (count toward Tier 1 in bootstrap): encoding-variant, exact, exact-match, not-an-error, tier-1, tier1. Residual labels stay the Stage 0 four genuine-misread, dropped-matra-nukta, reading-order-break, hallucinated-repeated-text.

## 2. Bootstrap Tier 1 rate (non-exact denominator)

### ALL

| Quantity | Value |
|----------|------:|
| denominator_non_exact | 280 |
| n_tier1_auto | 52 |
| n_tier2_auto | 0 |
| n_genuine_labeled | 13 |
| n_unreviewed | 215 |
| n_sample_adjudicated | 0 |
| n_sample_pending | 200 |
| n_sample_encoding_variant_labels | 0 |
| naive_tier1_rate_auto_only | 0.1857 |
| tier1_rate_ci95 | *withheld — no adjudicated labels* |

No adjudicated sample labels yet. Run `hand_review.py --queue data/predictions/adjudication_sample.jsonl` then re-run bootstrap. Naive auto-only Tier1 rate is reported above; CI withheld.

### tesseract

| Quantity | Value |
|----------|------:|
| denominator_non_exact | 152 |
| n_tier1_auto | 31 |
| n_tier2_auto | 0 |
| n_genuine_labeled | 13 |
| n_unreviewed | 108 |
| n_sample_adjudicated | 0 |
| n_sample_pending | 100 |
| n_sample_encoding_variant_labels | 0 |
| naive_tier1_rate_auto_only | 0.2039 |
| tier1_rate_ci95 | *withheld — no adjudicated labels* |

No adjudicated sample labels yet. Run `hand_review.py --queue data/predictions/adjudication_sample.jsonl` then re-run bootstrap. Naive auto-only Tier1 rate is reported above; CI withheld.

### surya

| Quantity | Value |
|----------|------:|
| denominator_non_exact | 118 |
| n_tier1_auto | 20 |
| n_tier2_auto | 0 |
| n_genuine_labeled | 0 |
| n_unreviewed | 98 |
| n_sample_adjudicated | 0 |
| n_sample_pending | 92 |
| n_sample_encoding_variant_labels | 0 |
| naive_tier1_rate_auto_only | 0.1695 |
| tier1_rate_ci95 | *withheld — no adjudicated labels* |

No adjudicated sample labels yet. Run `hand_review.py --queue data/predictions/adjudication_sample.jsonl` then re-run bootstrap. Naive auto-only Tier1 rate is reported above; CI withheld.

### paddleocr

| Quantity | Value |
|----------|------:|
| denominator_non_exact | 10 |
| n_tier1_auto | 1 |
| n_tier2_auto | 0 |
| n_genuine_labeled | 0 |
| n_unreviewed | 9 |
| n_sample_adjudicated | 0 |
| n_sample_pending | 8 |
| n_sample_encoding_variant_labels | 0 |
| naive_tier1_rate_auto_only | 0.1000 |
| tier1_rate_ci95 | *withheld — no adjudicated labels* |

No adjudicated sample labels yet. Run `hand_review.py --queue data/predictions/adjudication_sample.jsonl` then re-run bootstrap. Naive auto-only Tier1 rate is reported above; CI withheld.

## 3. Ranking test (Tier 1 normalisation)

Rates use all non-skipped predictions with GT (not the non-exact-only taxonomy CSV). raw_error = whitespace-normalized mismatch; tier1_error = normalize_tier1 mismatch.

### Engines (lower error rate = better)

| Engine | n | Raw error rate | Tier1 error rate |
|--------|---|---------------:|-----------------:|
| surya | 222 | 0.5045 | 0.4414 |
| tesseract | 180 | 0.7222 | 0.6722 |
| paddleocr | 10 | 0.9000 | 0.9000 |

- Order raw (best→worst): ['surya', 'tesseract', 'paddleocr']
- Order after Tier 1: ['surya', 'tesseract', 'paddleocr']
- **Order changed:** no

### Languages (lower error rate = better)

| Language | n | Raw error rate | Tier1 error rate |
|----------|---|---------------:|-----------------:|
| hindi | 250 | 0.5400 | 0.4960 |
| bengali | 120 | 0.6167 | 0.5167 |
| santhali | 42 | 1.0000 | 1.0000 |

- Order raw (best→worst): ['hindi', 'bengali', 'santhali']
- Order after Tier 1: ['hindi', 'bengali', 'santhali']
- **Order changed:** no

**Ranking stable:** Tier 1 shrinks error rates but does not reorder engines or languages on this corpus. The 20.4%-style rate claim stays a magnitude claim, not a leaderboard claim.

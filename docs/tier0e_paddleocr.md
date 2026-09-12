# Tier 0e — PaddleOCR full corpus taxonomy

**Status: computed** (2026-09-12, this laptop). Same scorer as Tesseract
and Surya: `python3 src/eval/error_taxonomy.py` (Tier 0 whitespace +
NFC inside `normalize_tier1`, then Tier 1, then Tier 2 ISO 15919, then
human notes). `paper/main.tex` was not edited.

Ground truth for *previous* Table 1 paddle row (n=10):
`docs/paper_defensibility_stats.md` Follow-Up 7. That n=10 was the only
Hindi rows without `skipped_reason` after OneDNN crashes. This run
**stripped** those crash rows, resumed PaddleOCR 3.7 with
`enable_mkldnn=False` (already in `run_baselines.py`), and re-scored.

## What was run

1. Backup of pre-strip jsonl:
   `data/predictions/paddleocr/_bak_before_tier0e/{hindi,bengali,kashmiri,santhali}.jsonl`
2. Dropped every line with `skipped_reason` (OneDNN
   `ConvertPirAttribute2RuntimeAttribute` errors). Hindi kept 10
   successful rows; other languages went to 0.
3. Resume: in-process `run_engine_over_language(..., per_image_timeout_seconds=None)`
   so one `PaddleOCR` pipeline is reused (spawn-timeout isolation was
   reloading weights every image, ~14s, and is the default CLI path).
   Same `run_paddleocr` / jsonl schema as Stage 0.
4. Languages and image set = the same `data/raw/{lang}/images/*.png`
   files Tesseract and Surya already have jsonl for (not a new corpus).

| Language | Images on disk | Paddle jsonl rows | `skipped_reason` | Empty `predicted_text` still scored |
|---|---:|---:|---:|---:|
| hindi | 120 | 120 | 0 | 3 |
| bengali | 60 | 60 | 0 | 10 |
| santhali | 200 | 200 | 0 | 5 |
| kashmiri | 40 | 40 | 0 | 1 |
| **total** | **420** | **420** | **0** | **19** empty strings counted as predictions |

Empty strings have `skipped_reason: null`, so `error_taxonomy.py`
**scores them** (almost always UNREVIEWED non-exact). They are not
dropped the way OneDNN crash rows were.

Lang codes (unchanged, `run_paddleocr`): hindi=`hi`, bengali=`en`
(no Bengali bundle in 3.7), santhali=`en`, kashmiri=`ar`.

## Table 1 shape (live recount)

Computation: `error_taxonomy.classify` over every prediction with no
`skipped_reason` and a GT id, same loop as
`src/analysis/paper_defensibility_stats.py` Follow-Up 7 (~lines 834–867).
Re-ran `python3 src/eval/error_taxonomy.py` after the jsonl fill.
Printed engine totals match the table.

| Engine | n | Exact-match | Tier 1 (of all) | Tier 1 among non-exact | TIER2 | GENUINE | UNREVIEWED |
|---|---:|---:|---:|---:|---:|---:|---:|
| tesseract (unchanged) | 180 | 28 (15.6%) | 31 (17.2%) | **20.4%** (31/152) | 0 | 13 | 108 |
| surya (unchanged) | 222 | 104 (46.8%) | 20 (9.0%) | **16.9%** (20/118) | 0 | 0 | 98 |
| **paddleocr, all 420 scored rows** | **420** | **11 (2.6%)** | **17 (4.0%)** | **4.2%** (17/409) | **0** | **0** | **392 (93.3%)** |

Tesseract n=180 is Hindi+Bengali only (Santhali/Kashmiri rows have
`skipped_reason`). Surya n=222 is Hindi+Bengali plus 42 Santhali without
skip. Paddle now **has no skips**, so n=420 is larger than both. That is
not a silent n change: garbage on Ol Chiki / Perso-Arabic / Bengali-via-`en`
is included.

### Same documents as Tesseract (Hindi+Bengali only)

| Slice | n | Exact-match | Tier 1 among non-exact |
|---|---:|---:|---:|
| paddle hi+bn | 180 | 11 (6.1%) | **10.1%** (17/169) |
| paddle hindi only | 120 | 11 (9.2%) | **15.6%** (17/109) |
| paddle bengali | 60 | 0 (0.0%) | 0/60 |
| paddle santhali | 200 | 0 | 0/200 |
| paddle kashmiri | 40 | 0 | 0/40 |

Every Paddle exact and every Paddle Tier 1 is **Hindi**. Bengali /
Santhali / Kashmiri contribute 0 EXACT and 0 TIER1.

Previous Table 1 paddle row (n=10, 0 exact, 1 Tier 1 / 10% of
non-exact) is superseded for engine-level claims by the Hindi n=120
row, or by hi+bn n=180 if the comparison must match Tesseract’s
language mix. Do not cite n=10 after this run.

TIER2 remains 0% after Tier 1 on this corpus (same finding as
Follow-Up 7).

## Caveats

- UNREVIEWED is 93.3% of all 420 Paddle rows (76.7% of Hindi). Residual
  labels are not a completed hand pass.
- In-process run disabled the per-image OS timeout (DECISIONS.md #34)
  for this fill only, so a hang would have blocked the batch. None did.
  Wall clock ~28.6 min (`terminals` elapsed_ms 1715256).
- This does **not** update `docs/paper_defensibility_stats.md`; that
  file remains the preprint’s recorded Table 1 until you regenerate it.

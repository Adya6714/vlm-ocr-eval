# Probe 6 — synthetic Claim B vs real Tier C

**Generated:** 2026-09-03  
**Seeds:** []  
**n_boot:** 10000  
**δ (SEOI):** 0.05  
**Scope:** hindi/natural only; Tier C plain+degraded+blank. Full Probe 6 (Tier B sweep, handwriting anecdote, held-out synthetic pages 100–109, multi-system gaps) deferred — see § Future work and DECISIONS.md #58.

---

## 0. Held-out validity (data leakage)

Training manifests under `data/manifests` were checked against `data/raw/hindi/images` (120 files).

| Manifest | n image_path |
|----------|--------------|
| `hindi_natural.jsonl` | 2538 |
| `hindi_flattened.jsonl` | 2707 |
| `hindi_inverted.jsonl` | 2872 |

**Confirmed: 0 overlaps.** Manifest paths are renderer line crops (`data/cache/line_crops/...`); Tier C images are GlotOCR-bench via `fetch_glotocr.py` under `data/raw/hindi/images/`. The instrument never trained on these files — Probe 6 is a valid held-out test.

*No probe6_synthetic_real_hindi_seed*.jsonl on disk yet.*

## Future work (full Probe 6 — not built)

- Tier B degradation sweep on synthetic renders
- Handwriting anecdote (15–20 lines, 2–3 writers, qualitative)
- Held-out synthetic pages 100–109 (DECISIONS.md #45) for a clean accuracy-gap estimate
- Multi-system gaps (Tesseract / Surya / PaddleOCR / instrument)
- The previous metric-only aggregator API in this file's history is superseded for the paper-deadline scope

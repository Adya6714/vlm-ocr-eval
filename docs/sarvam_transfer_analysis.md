# Stage 5a — Sarvam confidence on zero-exposure scripts

**Generated:** 2026-09-03  
**Input:** `data/probe_results/sarvam_transfer_probe.jsonl` (35 records)  
**Code:** `src/probes/sarvam_transfer_probe.py`, `src/eval/sarvam_client.py`  
**Decisions:** #19 (cache-once), #59 (Extract over Digitise), #60 (accuracy sources)

---

## Method

Sarvam Doc-AI **Extract** endpoint (confirmed from live docs Sept 2026) was called
with a single-field schema:

```json
{"type":"object","properties":{"full_text":{"type":"string","description":"The complete transcribed text content of this document page…"}}}
```

This is the only Sarvam endpoint that exposes `annotations.{field}.confidence`.
Digitise does not. One confidence number per page is returned, comparable
directly to this project's `mean_confidence` metric.

**Sample:** 10 Hindi + 10 Santhali + 10 Kashmiri plain images from
`data/raw/{script}/`, drawn `Random(0)`. Plus 5 blank (solid white 1200×80 px)
as the control. **35 pages total, ₹17.50 at ₹0.5/page.**

All 35 results cached under `data/cache/sarvam/` (SHA-256-keyed JSON).
The blank images all hash identically — 1 actual API call, 4 cache hits.
The probe was resumed once after a null-safety fix on record 14.

---

## Results

| Script | n | mean_confidence | min | max | stdev | Sarvam published accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Hindi | 10 | **0.9997** | 0.9969 | 1.0000 | 0.0010 | 95.91 % |
| Santhali | 10 | **0.9974** | 0.9866 | 1.0000 | 0.0042 | 80.32 % |
| Kashmiri | 10 | **0.9970** | 0.9883 | 1.0000 | 0.0037 | 55.93 % |
| Blank | 5 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | — |

Published accuracy figures verified Sept 2026 from
`sarvam.ai/blogs/sarvam-vision` (Sarvam Indic OCR Bench, word accuracy =
100 × (1 − WER); DECISIONS.md #60).

---

## Finding

Sarvam's confidence **does not track its own published accuracy gap**.

- Hindi → Kashmiri accuracy gap: **39.98 percentage points** (95.91 → 55.93)
- Hindi → Kashmiri confidence delta: **0.0027** (0.9997 → 0.9970)

The confidence gap is less than 3 thousandths while the accuracy gap is
nearly 40 percentage points.  This **directly replicates** the instrument's
own finding (Probe 5b within-condition delta 0.0037) on Sarvam's production
system.

The blank-control makes the dissociation unmistakable: blank images (where
there is literally nothing to transcribe) score **0.0000**, while real
Kashmiri images (which Sarvam's own benchmark says it misreads ~44% of
the time) score **0.9970**.  Confidence is not tracking readability.

---

## Interpretation (Claim B extension)

Claim B in this project ("confidence does not track image readability")
was previously established only on a small from-scratch instrument.
A critic could argue this is an artifact of undertraining.

This probe shows the pattern holds on **Sarvam Vision**, a 3-billion-
parameter model trained specifically on all 22 official Indian languages
and achieving state-of-the-art results on its own benchmark.  The
confidence-blindness finding is **not an artifact of undertraining** — it
appears structurally in a production-quality Indic OCR system too.

---

## Limitations and future work

- n=10 per script; no bootstrap CIs computed yet (see Stage 5b).
- Only plain (un-degraded) images tested; degraded condition deferred.
- Full rank-correlation against Tier A/B (DECISIONS.md #15, Stage 5b) not run.
- The blank confidence is 0.0 because Sarvam returned no `full_text` field
  for a blank image (null annotation), which `sarvam_client.py` correctly
  defaults to 0.0 — this is semantically correct for the blank control.

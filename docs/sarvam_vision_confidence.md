# Sarvam Vision: confidence versus published Indic OCR accuracy

This note is for researchers and engineers working on **Sarvam Vision** and **Doc-AI** (digitise / extract). It states why this repository exists next to a production Indic OCR stack, what we measured on **Extract**, and which probes in the repo are usable if confidence is treated as a review signal.

The preprint [*Reading Without Looking*](../paper/main.pdf) studies a from-scratch instrument. This file is the **Sarvam-facing** evaluation: public bench numbers vs Extract confidence, plus the mechanistic work that makes a flat confidence curve interpretable.

Measured cells and cache policy originally recorded in the Stage 5a write-up live here (single document).

---

## Motivation

Sarvam publishes Indic OCR **word accuracy** that already varies sharply by language (Sarvam Indic OCR Bench, word accuracy \(= 100\times(1-\mathrm{WER})\)). A deployed Extract (or similar) path also returns a **confidence**. If that scalar is used to decide what a human should check, it should move with the published hardness of the language—or at least with whether the page contains text at all.

Two questions follow:

1. On Extract, does mean confidence track the **published** Hindi–Kashmiri accuracy gap?
2. If it does not, is that an artifact of “the small research model was undertrained,” or a **structural** property of decoder-style confidence (high whenever the string looks like language)?

A live API cannot answer (2): it does not expose encoder memory, per-step softmaxes, or a known-empty Indic pretraining history. This repository trains a ~19.6M-parameter encoder–decoder **from scratch**, then runs blank pages, unseen scripts, encoder ablation, and teacher-forced likelihood **by generation position**. Position 0 has no generated prefix, so mass on the correct first grapheme has to come from the image.

That instrument is the prior for reading Extract: we already know a reader can sit near confidence 1 with grapheme CER ≈ 1. The API study then asks whether Sarvam’s own published gap appears in Extract confidence.

---

## Extract vs the Indic OCR Bench

**Endpoint.** Doc-AI **Extract** (not Digitise). Extract exposes `annotations.{field}.confidence`. Digitise does not. Client: `src/eval/sarvam_client.py`. Probe: `src/probes/sarvam_transfer_probe.py`.

**Schema.** Single field `full_text`, so one confidence per page, comparable to the instrument’s mean max-softmax.

**Sample.** 10 Hindi + 10 Santhali + 10 Kashmiri plain pages (`data/raw/{script}/`, `Random(0)`), plus 5 blank white images. **35 pages.** Responses cached under `data/cache/sarvam/` (SHA-256 of the request). Blank images share one hash (one live call, four cache hits). Do not re-call the API in a loop when the cache already holds the page.

Published accuracies below were checked against [Sarvam’s public Vision write-up](https://sarvam.ai/blogs/sarvam-vision) (September 2026).

| Script | n | Mean Extract confidence | Published word accuracy |
|---|---:|---:|---:|
| Hindi | 10 | **0.9997** | 95.91% |
| Santhali | 10 | **0.9974** | 80.32% |
| Kashmiri | 10 | **0.9970** | 55.93% |
| Blank | 5 | **0.0000** | — |

- Hindi → Kashmiri **accuracy** gap: **39.98 percentage points** (95.91 → 55.93).
- Hindi → Kashmiri **confidence** Δ: **0.0027** (0.9997 → 0.9970).

Blank pages score 0.0000 because Extract returned no `full_text` (null annotation), mapped to 0 in the client. The meter can go to zero. It does not on Kashmiri, which the same bench reports as misread a large fraction of the time.

This is **not** a claim that Extract is the 19.6M instrument. It is the same dissociation—tens of points of published accuracy, thousandths of confidence—on Sarvam’s production API.

**Limits.** n = 10 per script; no bootstrap CI; plain images only; full rank-correlation of confidence against page accuracy (Stage 5b) is specified and not run. jsonl: `data/probe_results/sarvam_transfer_probe.jsonl`.

---

## What in this repo is for a Vision / Doc-AI team

| Need | Where |
|---|---|
| Encoding-fair scoring before residual error | `src/eval/equivalence_tables.py`, Tier 2 transliteration |
| Blank, unseen script, encoder ablation, position-wise *p*(GT) | `src/probes/` |
| Position-0 argument and figures | `paper/main.pdf`, [project page](https://adya6714.github.io/vlm-ocr-eval/) |
| Extract vs published bench | this file |
| Inference-only follow-ups (mismatch pairing, attention norms, Surya control) | [`remaining_measurements.md`](remaining_measurements.md) |

Instrument tables: [`paper_defensibility_stats.md`](paper_defensibility_stats.md), [`RESULTS.md`](RESULTS.md).

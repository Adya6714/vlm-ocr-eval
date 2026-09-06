# Confidence as a review signal for Indic document AI

This note is for teams who ship **document OCR or field extraction** and expose a **confidence** used to decide what a human should check—especially across Indian languages, where published accuracy already varies a great deal by script.

The preprint [*Reading Without Looking*](../paper/main.pdf) studies a small from-scratch reader. It does not discuss a vendor product. The measurements below are a **separate, cached API evaluation** in this repository. Together they answer two questions that a production vision stack actually faces: (1) can decoder/page confidence fail as a visual-certainty signal even when the architecture is an encoder–decoder over pages, and (2) does that failure appear on a live Indic document endpoint whose own public benchmark already reports a large language gap?

## Why an instrument, not only an API score

A production API returns text and a scalar. It does not return encoder memory, per-step softmaxes, or a guarantee that “this checkpoint has never seen Ol Chiki.” If confidence is ~1 on a language the public bench calls hard, you cannot tell from the API alone whether that is **undertraining**, a **stuck meter**, or a **language prior** that does not need the page.

This repo trains a ~19.6M-parameter encoder–decoder with **no Indic pretraining**, then runs probes that a closed model cannot: blank and noise pages, unseen scripts, encoder-memory ablation, and teacher-forced log *p*(ground truth) **by generation position**. Position 0 is the clean test: no generated prefix, so any mass on the correct first grapheme must come from the image.

On that instrument, held-out grapheme CER is near 1 on both text-bearing and blank pages, while confidence stays near ceiling. Ablating the encoder barely moves the peak. That is the mechanistic prior for reading an API result.

Protocol and tables for the instrument: [`paper_defensibility_stats.md`](paper_defensibility_stats.md), [`RESULTS.md`](RESULTS.md).

## API evaluation (Extract)

Sarvam Doc-AI **Extract** is the endpoint that returns per-field `confidence` (Digitise does not). We called it on 35 cached pages (10 Hindi, 10 Santhali, 10 Kashmiri, 5 blank). Published word accuracies on the Sarvam Indic OCR Bench (word accuracy = \(100 \times (1-\mathrm{WER})\), figures checked against Sarvam’s public write-up in September 2026) are Hindi **95.91** and Kashmiri **55.93**—a **39.98 percentage-point** gap. Mean Extract confidence on the same pair is Hindi **0.9997** vs Kashmiri **0.9970** (Δ **0.0027**). Blank pages score **0.0000**, so the channel can go to zero; it does not on the language the bench marks as hard.

Full protocol, cache policy, and limits (n = 10 per script; rank-correlation not run): [`sarvam_transfer_analysis.md`](sarvam_transfer_analysis.md). Client: `src/eval/sarvam_client.py`. Probe: `src/probes/sarvam_transfer_probe.py`. Reuse `data/cache/sarvam/` rather than repeating live calls.

This is not an identification of Extract with the 19.6M instrument. It is the same **dissociation**—large published accuracy gap, negligible confidence gap—on a production Indic document API.

## What in this repo is reusable

| Need | Use |
|---|---|
| Encoding-fair scoring before residual error | `src/eval/equivalence_tables.py`, Tier 2 transliteration |
| Blank / unseen-script / ablation / position-wise likelihood | `src/probes/` |
| Public figures and the position-0 argument | `paper/main.pdf`, project page Preprint tab |
| Extract vs published bench | this note + `sarvam_transfer_analysis.md` |
| Inference-only follow-ups (mismatch pairing, attention norms, Surya control) | [`remaining_measurements.md`](remaining_measurements.md) |

A larger rank-correlation of Extract confidence against page-level accuracy (Stage 5b) is specified and not run. The 35-page probe is a minimal existence check, not a substitute for that study.

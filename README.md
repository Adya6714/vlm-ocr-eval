# vlm-ocr-eval

**A diagnostic instrument for Indic document OCR, not an OCR system.**

When a document VLM misreads Devanagari or Bengali, *why* — and can we tell
“cannot see the glyph” from “confidently guessing from the language prior”?
That needs a model you own end to end. This repo builds a small **instrument**
with zero Indic pretraining exposure, runs probes against it, and (planned)
transfers findings to a production API.

**Live walkthrough:** [https://adya6714.github.io/vlm-ocr-eval/](https://adya6714.github.io/vlm-ocr-eval/)
(`site/index.html` — Walkthrough / Paper modes, repo tour, findings).

Teaching rebuild path: **[`BOOK.md`](./BOOK.md)** (concept → code → evidence).
Spec / why / tasks: `IMPLEMENTATION.md`, `DECISIONS.md`, `TODO.md`, `AGENTS.md`.

---

## Layers (required / extension / research)

| Layer | What | Where | BOOK |
|---|---|---|---|
| **Required** | Taxonomy, renderer dial, instrument, probes | `src/eval/`, `src/renderer/`, `src/models/instrument/`, `src/probes/` | Ch. 1–3, 7 |
| **Extensions** | Resume/Colab export, hand-review assist, line manifests, smoke makefile, aggregation | `run_baselines.py`, `hand_review_assist.py`, `export_manifest_scaled.py`, `makefile`, `src/analysis/` | Ch. 1–2, App. F |
| **Research / deferred** | Demo LoRA, structure metrics, RLVR, Sarvam transfer, cascade, Probe 5b | mostly unbuilt; see Future Work | Ch. 4–6, 8–9 |

---

## Status

| Stage | Status |
|---|---|
| 0 Error taxonomy (Tesseract / Surya / PaddleOCR) | Built; run on real data — [BOOK Ch. 1](./BOOK.md#chapter-1--measuring-correctness-is-its-own-hard-problem) |
| 1 Controlled renderer (glyph-frequency dial) | Built; Tier A/B verified — [Ch. 2](./BOOK.md#chapter-2--turning-text-back-into-pixels-on-purpose) |
| 2a Instrument (from-scratch encoder/decoder) | Built (~19.5M); train/probe artifacts may live off-tree — [Ch. 3](./BOOK.md#chapter-3--learning-to-read-from-nothing-the-instrument) |
| 2b Demo (LoRA VLM) | **Not built** — [Ch. 4](./BOOK.md#chapter-4--teaching-an-existing-model-a-new-trick-cheaply-demo) |
| 3 Reading-order / table-binding metrics | Not built — [Ch. 5](./BOOK.md#chapter-5--where-does-the-text-go-on-the-page) |
| 4 Probe suite | Code built; 5b not built — [Ch. 7](./BOOK.md#chapter-7--what-does-the-model-actually-know-the-probe-suite) |
| 5 Sarvam transfer | Not built — [Ch. 8](./BOOK.md#chapter-8--why-you-cant-learn-everything-from-an-api) |
| 6 Triage cascade | Not built — [Ch. 9](./BOOK.md#chapter-9--when-to-trust-a-machine-and-when-to-escalate) |

---

## Headline results (reproduce in BOOK Appendix E)

- **Stage 0 (measured):** of Tesseract predictions that were not exact match, **~20.4%** were Tier 1 encoding variants (not genuine misreads). Large `UNREVIEWED` remains — report is real, not final.
- **Stage 1 (measured):** glyph-frequency modes hit \(\mathrm{TV}\le 0.08\) on the Hindi GT slice (natural 0 / flat ≈0.047 / inv ≈0.005); Tier A pages render well under 1 s.
- **Probes 3 & 5 (reported in prior Colab write-up; jsonl not in this checkout):** mean confidence ≈0.99 on real, blank, and noise; high-confidence bucket accuracy ≈0.10 — confidence not grounded in the image at that checkpoint. Re-run commands: BOOK [Appendix E](./BOOK.md#appendix-e--reproduce-every-headline-number).

---

## Core idea (three sentences)

1. **Not every diff is an error** — Tier 1 encoding / Tier 2 phonetic equivalence before “genuine.”
2. **Exposure is a dial** — `natural` / `flattened` / `inverted` glyph-frequency modes, volume-matched.
3. **Confidence is the measurand** — blank/noise controls and calibration curves, not CER alone.

Pipeline sketch: GlotOCR → Stage 0 taxonomy; same corpus → Stage 1 renderer → line manifests → instrument (3×3) → probes. Details and mermaid: [BOOK requirement map](./BOOK.md#requirement--product-map).

---

## Running it

Heavy compute: Colab, one `--data-root`. Architecture without findings: `make smoke-test`.

```bash
python src/data_pipeline/export_manifest_scaled.py --script hindi --pages-per-mode 100
python src/models/instrument/train.py \
  --manifest data/manifests/hindi_natural.jsonl \
  --output-root <checkpoints> --condition natural --seed 0
python src/eval/error_taxonomy.py
pytest -q
```

---

## Docs map

| File | Role |
|---|---|
| [Live site](https://adya6714.github.io/vlm-ocr-eval/) | Interview walkthrough (GitHub Pages ← `site/`) |
| [`BOOK.md`](./BOOK.md) | Teaching narrative — rebuild from first principles |
| [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) | Module spec + status |
| [`DECISIONS.md`](./DECISIONS.md) | Numbered design choices |
| [`TODO.md`](./TODO.md) | Ordered work |
| [`AGENTS.md`](./AGENTS.md) | Agent workflow |
| [`COLAB_RUNS.md`](./COLAB_RUNS.md) | What ran where |

---

## Future work (deferred on purpose)

**2b Demo, RLVR, Sarvam transfer (~200 cached pages), cascade (router quality not cost savings), Probe 5b (Santhali/Kashmiri zero-shot floor)** — reasoning in [BOOK Ch. 4–6, 8–9](./BOOK.md) and README history in git; not claimed as built.

**More compute** would lengthen training, enlarge corpus, and full-sample probes so “undertrained” separates from “structurally ungrounded confidence.” Design stays; weight of the numbers changes.

---

This repository is a diagnostic apparatus and a methodology argument — not an OCR product, benchmark, or dataset release. Unbuilt stages and provisional numbers are labeled as such.

# vlm-ocr-eval

**A diagnostic instrument for Indic document OCR, not an OCR system.**

When a document VLM misreads Devanagari or Bengali, *why* — and can we tell
“cannot see the glyph” from “confidently guessing from the language prior”?
That needs a model you own end to end. This repo builds a small **instrument**
with zero Indic pretraining exposure, runs probes against it, and (planned)
transfers findings to a production API.

**Live walkthrough:** [https://adya6714.github.io/vlm-ocr-eval/](https://adya6714.github.io/vlm-ocr-eval/)
(repo-root [`index.html`](./index.html) — Walkthrough / Paper modes, repo tour, findings).

Teaching rebuild path: **[`BOOK.md`](./BOOK.md)** (concept → code → evidence).
Spec / why / tasks: `IMPLEMENTATION.md`, `DECISIONS.md`, `TODO.md`, `AGENTS.md`.

---

## Layers (required / extension / research)

| Layer | What | Where | BOOK |
|---|---|---|---|
| **Required** | Taxonomy, renderer dial, instrument, probes | `src/eval/`, `src/renderer/`, `src/models/instrument/`, `src/probes/` | Ch. 1–3, 7 |
| **Extensions** | Resume/Colab export, hand-review assist, line manifests, smoke makefile, aggregation | `run_baselines.py`, `hand_review_assist.py`, `export_manifest_scaled.py`, `makefile`, `src/analysis/` | Ch. 1–2, App. F |
| **Research / deferred** | Demo LoRA, structure metrics, RLVR, Sarvam transfer, cascade | mostly unbuilt; see Future Work | Ch. 4–6, 8–9 |

---

## Status

| Stage | Status |
|---|---|
| 0 Error taxonomy (Tesseract / Surya / PaddleOCR) | Built; run on real data — [BOOK Ch. 1](./BOOK.md#chapter-1--measuring-correctness-is-its-own-hard-problem) |
| 1 Controlled renderer (glyph-frequency dial) | Built; Tier A/B verified — [Ch. 2](./BOOK.md#chapter-2--turning-text-back-into-pixels-on-purpose) |
| 2a Instrument (from-scratch encoder/decoder) | Built (~19.5M); Hindi Probe 1/3/5/5b artifacts in `data/probe_results/` — [Ch. 3](./BOOK.md#chapter-3--learning-to-read-from-nothing-the-instrument) |
| 2b Demo (LoRA VLM) | **Not built** — [Ch. 4](./BOOK.md#chapter-4--teaching-an-existing-model-a-new-trick-cheaply-demo) |
| 3 Reading-order / table-binding metrics | Not built — [Ch. 5](./BOOK.md#chapter-5--where-does-the-text-go-on-the-page) |
| 4 Probe suite | **Mixed** — see probe rows below — [Ch. 7](./BOOK.md#chapter-7--what-does-the-model-actually-know-the-probe-suite) |
| → Probe 1 (exposure FE) | Ran; headline β withheld (flattened/inverted ~0% line acc) |
| → Probe 2 (confusion / p(true)) | **BUILT — BLOCKED** (code + tests; Colab inference pending) |
| → Probe 3 / 3b (blank + training curve) | **BUILT — VERIFIED** (jsonl/json in `data/probe_results/`) |
| → Probe 5 / 5b (calibration + zero-shot floor) | **BUILT — VERIFIED** (5b: 720 records, 3 seeds) |
| → Attention ablation (Claim B mechanism) | **BUILT — BLOCKED** (code + tests; Colab inference pending) |
| → Probe 6 (Tier C vs synthetic, paper scope) | **BUILT — BLOCKED** (code + tests + leakage check; Colab inference pending) |
| 5 Sarvam transfer | Not built — [Ch. 8](./BOOK.md#chapter-8--why-you-cant-learn-everything-from-an-api) |
| 6 Triage cascade | Not built — [Ch. 9](./BOOK.md#chapter-9--when-to-trust-a-machine-and-when-to-escalate) |

Numbers and acceptance criteria: [`IMPLEMENTATION.md`](./IMPLEMENTATION.md). Blocked items are not verified findings.

---

## Headline results (reproduce in BOOK Appendix E)

- **Stage 0 (measured):** of Tesseract predictions that were not exact match, **20.4%** were Tier 1 encoding variants (not genuine misreads). Tier 2 validation is a **complete 38-pair** hand-checked set (23 positive / 15 negative+boundary; `docs/tier2_validation.md`, DECISIONS.md #54) — not a stub; corpus TIER2 rate after Tier 1 is currently 0%. Large `UNREVIEWED` remains — report is real, not final. Stratified adjudication sample (n=200) + bootstrap/ranking: `src/analysis/adjudication_sample.py`, `docs/adjudication_analysis.md` (DECISIONS.md #55). Tier 1 does **not** reorder engines/languages on this corpus; bootstrap CI waits on human `--queue` labels.
- **Stage 1 (measured):** glyph-frequency modes hit \(\mathrm{TV}\le 0.08\) on the Hindi GT slice (natural 0 / flat ≈0.047 / inv ≈0.005); Tier A pages render well under 1 s.
- **Probe 5b (BUILT — VERIFIED; 3 seeds, 720 records in `data/probe_results/probe5b_hindi_natural_seed{0,1,2}.jsonl`):** between-condition range of across-seed mean confidence is **0.0037**, smaller than within-condition across-seed SD (**0.0043** hindi, **0.0066** blank). Script substitution **360/360** on Santhali+Kashmiri (zero graphemes of the script in the image). **Methods-integrity note:** the seed-0 Kashmiri-vs-Hindi Bonferroni “significance” pass is **retracted** — it does not replicate across seeds (DECISIONS.md #53; `docs/probe5b_analysis.md`, `docs/statistical_repair.md`).
- **Probe 3b (BUILT — VERIFIED; 3 seeds, `probe3_curve_hindi_natural_seed{0,1,2}.json`):** real−blank confidence gap is **indistinguishable from zero** across seeds (sign-flips at 4/5 steps; |SD| > |mean| at 4/5 steps) while loss falls ~**18×** (3.210 → 0.182). Undertraining and ungrounded confidence are **not** competing explanations — confidence rises as loss falls (`docs/probe3_curve_analysis.md`).
- **Probes 3 & 5 (BUILT — VERIFIED; jsonl committed under `data/probe_results/`):** mean confidence stays near ceiling on real / blank / noise; calibration remains poorly grounded at these checkpoints. Re-run commands: BOOK [Appendix E](./BOOK.md#appendix-e--reproduce-every-headline-number).

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
  --script hindi \
  --output-root <checkpoints> --condition natural --seed 0
python src/eval/error_taxonomy.py
pytest -q
```

---

## Docs map

| File | Role |
|---|---|
| [Live site](https://adya6714.github.io/vlm-ocr-eval/) | Interview walkthrough (GitHub Pages ← repo-root `index.html`) |
| [`BOOK.md`](./BOOK.md) | Teaching narrative — rebuild from first principles |
| [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) | Module spec + status |
| [`DECISIONS.md`](./DECISIONS.md) | Numbered design choices |
| [`TODO.md`](./TODO.md) | Ordered work |
| [`AGENTS.md`](./AGENTS.md) | Agent workflow |
| [`COLAB_RUNS.md`](./COLAB_RUNS.md) | What ran where |
| [`docs/probe5b_analysis.md`](./docs/probe5b_analysis.md) | Probe 5b claim-facing write-up |
| [`docs/probe3_curve_analysis.md`](./docs/probe3_curve_analysis.md) | Probe 3b curve write-up |
| [`docs/statistical_repair.md`](./docs/statistical_repair.md) | Bootstrap / TOST / Kashmiri retraction |

---

## Future work (deferred on purpose)

**2b Demo, RLVR, Sarvam transfer (~200 cached pages), cascade (router quality not cost savings)** — reasoning in [BOOK Ch. 4–6, 8–9](./BOOK.md); not claimed as built.

**Probe 5b is built** (above). Still blocked on Colab checkpoints (code ready, not VERIFIED): attention ablation, Probe 2 GT-aligned confusion, Probe 6 Tier C paper scope — see `IMPLEMENTATION.md`.

**More compute** would lengthen training, enlarge corpus, and finish blocked inference so “undertrained” separates from “structurally ungrounded confidence” with full mechanism probes. Design stays; weight of the numbers changes.

---

This repository is a diagnostic apparatus and a methodology argument — not an OCR product, benchmark, or dataset release. Unbuilt stages and provisional / blocked numbers are labeled as such.

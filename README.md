# vlm-ocr-eval

A **from-scratch Indic OCR instrument** and a **probe suite**, not a shipping OCR engine.

Autoregressive readers emit a token and a confidence at every step. Production systems use that confidence to decide what a human should check. This repo asks whether that number tracks *whether the image supports the text*, or only *whether the generated string looks like the training language*.

The paper is [**Reading Without Looking**](paper/main.pdf) (`paper/main.tex`). The instrument is a ~19.5M-parameter encoder–decoder trained with **no Indic pretraining**. On held-out real Hindi scans it does **not** read (grapheme CER near 1 on both real pages and blank white). Decoder confidence stays near ceiling anyway. Position 0 is the informative test: there is no prefix, so any probability on the correct first symbol has to come from the image. It does not.

Numbers in this README stay qualitative on purpose. Trace every figure to jsonl and code in [`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/paper_defensibility_stats.md`](docs/paper_defensibility_stats.md). Do not copy tables from chat.

**Site:** [adya6714.github.io/vlm-ocr-eval](https://adya6714.github.io/vlm-ocr-eval/) (repo-root [`index.html`](index.html) — preprint tab vs Extract-confidence audit).  
**Teaching path:** [`BOOK.md`](BOOK.md) (concept → files → what was measured).

---

## What lives where

```
paper/                 Preprint (compile main.tex; figures in paper/figures/)
index.html             GitHub Pages site
BOOK.md                First-principles chapters
IMPLEMENTATION.md      Module spec and [x]/[~]/[!] status
DECISIONS.md           Numbered design choices (do not re-litigate silently)
TODO.md / AGENTS.md    Work order and agent rules
src/eval/              Stage 0 engines, Tier 1/2 scoring, taxonomy
src/renderer/          Controlled line rendering (glyph-frequency dial)
src/data_pipeline/     GlotOCR fetch, page → line-crop manifests
src/models/instrument/ Encoder, decoder, train, generate, grapheme tokenizer
src/probes/            Blank/noise, calibration, ablation, GT-likelihood, …
src/analysis/          Stats, figures, claim-facing markdown in docs/
data/manifests/        Training line lists (committed)
data/probe_results/    Probe jsonl (committed; small)
data/raw/              GlotOCR images + GT (gitignored)
data/predictions/      Tesseract/Surya/Paddle jsonl (gitignored)
checkpoints/           Weights (gitignored; Colab/Drive)
docs/                  Analyses, training_config.md, remaining_measurements.md
```

Heavy runs are Colab T4 (`COLAB_RUNS.md`). This laptop is where results **land**.

---

## How the pieces connect

1. **Score fairly.** Indic OCR has many valid encodings of the same reading. `src/eval/equivalence_tables.py` (Tier 1) and `transliteration_equivalence.py` (Tier 2) run **before** residual error is counted. Engine table: pooled prediction rows, lower bounds because of UNREVIEWED mass — see the preprint and `docs/paper_defensibility_stats.md` Follow-Up 7.
2. **Control exposure.** Training text is resampled into `natural` / `flattened` / `inverted` glyph-frequency modes (`src/renderer/`), then cropped to 70 px lines (`src/data_pipeline/export_manifest_scaled.py`).
3. **Own the weights.** `src/models/instrument/train.py` trains one Hindi/natural checkpoint per seed (0, 1, 2). Flattened/inverted runs exist as a failed exposure experiment, not extra seeds of a reader.
4. **Open the decoder.** Probes in `src/probes/` measure blank vs real vs unseen scripts, encoder-memory zeroing, teacher-forced log p(GT) by position, calibration. Analysis scripts write `docs/*_analysis.md`.

A production LoRA demo, reading-order metrics, RLVR, and a triage cascade are **specified and deferred** (`IMPLEMENTATION.md` Stages 2b–6). They are not claimed as built.

---

## Start here

| If you want… | Open |
|---|---|
| The scientific claim | [`paper/main.pdf`](paper/main.pdf) · [`paper/README.md`](paper/README.md) |
| A visual tour | [GitHub Pages](https://adya6714.github.io/vlm-ocr-eval/) |
| Why the code looks like this | [`BOOK.md`](BOOK.md) |
| Exact hyperparameters / eval provenance | [`docs/training_config.md`](docs/training_config.md) |
| What is still unrun (no checkpoints in git) | [`docs/remaining_measurements.md`](docs/remaining_measurements.md) |
| Reproduce a headline number | [`docs/RESULTS.md`](docs/RESULTS.md) · BOOK Appendix E |

Architecture-only smoke (no findings): `make smoke-test`. Tests: `pytest -q`.

Train (needs a GPU or patience; checkpoints are not in git):

```bash
python src/models/instrument/train.py \
  --manifest data/manifests/hindi_natural.jsonl \
  --script hindi --condition natural --seed 0 \
  --output-root checkpoints
```

Paper figures:

```bash
PYTHONPATH=src/eval python3 src/analysis/make_paper_figures.py \
  --results-root data/probe_results \
  --out-dir docs/figures --paper-dir paper/figures
```

Compile the preprint from `paper/` (`pdflatex`+`bibtex` or `tectonic -X compile paper/main.tex`).

---

## What this is not

Not a dataset release (raw scans stay gitignored). Not a leaderboard. Not Sarvam’s product page — the paper does not name that API; the site’s second tab is a separate Extract-confidence question on the same URL (`DECISIONS.md` #69).

If you are contributing with an agent, read `AGENTS.md` first.

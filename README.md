# vlm-ocr-eval

Diagnostic evaluation of **decoder confidence** in Indic document OCR.

Production systems use per-token or per-page confidence to decide what a human should review. This repository asks whether that signal tracks **whether the image supports the text**, or only **whether the generated string looks like language**.

| | |
|---|---|
| Preprint | [*Reading Without Looking*](paper/main.pdf) · [source](paper/main.tex) |
| Project page | [adya6714.github.io/vlm-ocr-eval](https://adya6714.github.io/vlm-ocr-eval/) |
| Measurements | [`docs/RESULTS.md`](docs/RESULTS.md) · [`docs/paper_defensibility_stats.md`](docs/paper_defensibility_stats.md) |
| Training setup | [`docs/training_config.md`](docs/training_config.md) |
| Sarvam Vision / Extract | [`docs/sarvam_vision_confidence.md`](docs/sarvam_vision_confidence.md) |

The instrument is a ~19.6M-parameter encoder–decoder trained **from scratch** (no Indic pretraining) on rendered Hindi line crops. It is a measurement device, not a deployed OCR engine.

---

## Overview

Autoregressive OCR emits a symbol and a confidence at every step. If that confidence stays high on blank pages, unseen scripts, or pages the model cannot read, review queues based on it are miscalibrated.

Indic documents add two complications: multiple valid Unicode encodings of the same reading (so exact-match error rates are noisy), and published systems that fail **non-silently**—high confidence when the script in the image is one the model cannot read.

A closed API cannot explain *why*. This project therefore:

1. Normalizes encodings **before** counting residual errors (Tier 1 / Tier 2).
2. Trains a small reader with a known, empty Indic pretraining history.
3. Probes blank input, unseen scripts, encoder ablation, and teacher-forced likelihood **by generation position**.

The informative test is **position 0**: there is no generated prefix, so any probability on the correct first grapheme has to come from the image.

---

## Findings

Headline numbers are computed from committed probe outputs in `data/probe_results/`. Full tables and code paths: [`docs/paper_defensibility_stats.md`](docs/paper_defensibility_stats.md), [`docs/RESULTS.md`](docs/RESULTS.md).

- **The instrument does not read its held-out images.** Grapheme CER is near 1 on both text-bearing Hindi pages and blank white images (pooled ~0.985 vs ~0.949). After clustering by seed the difference is not significant. Evaluation images are GlotOCR **renders**, not photographs.
- **Confidence stays near ceiling** on text-bearing Hindi, blank pages, Ol Chiki, and Perso-Arabic (condition means within ~0.005).
- **Position 0:** geometric-mean *p*(ground truth) is on the order of 10⁻¹¹ while self-generated max-softmax is ~0.90. The comparison is to a **text-only grapheme *n*-gram**, not to a uniform vocabulary prior. Ground truth is never the argmax at position 0 (0/180 sequences).
- **Encoder ablation** changes mean confidence by ~−0.003. About 8% of decoding steps flip the argmax and account for ~97% of the KL; agreeing steps keep a near-unit peak.
- **Mid-sequence** teacher-forced log *p*(GT) sits in the same band as a 4- to 5-gram grapheme language model. Full-softmax KL vs that 5-gram is low there (~0.27–0.32 nats) with ~91–94% argmax agreement; positions 0, 1, and 40+ diverge (`docs/ngram_kl_argmax.md`).
- **Noise and patch-scrambled** images do not open a visual gap: whole-sequence mean log *p*(GT) stays within 0.05 nats of text-bearing and blank (`docs/tier0d_noise_scrambled.md`).
- The instrument was trained on the **full** 2,538-line manifest (19 of 60 evaluation strings appear as training lines). Probe 5's AUROC 0.838 is **in-training-manifest only**: every synthetic eval string is in that file (`docs/memorisation_split.md`).
- **PaddleOCR** as an instrument-matched positive control is **not viable** (CTC rec head, not autoregressive; `docs/paddleocr_feasibility.md`). Table 1 still uses it as an off-the-shelf engine (n=420).

Figures 1–4: `paper/figures/` (PDF) and `docs/figures/` (PNG).

---

## Method

```
GlotOCR pages
    ├── baseline engines (Tesseract, Surya, PaddleOCR)
    │       encoding-aware scoring → residual taxonomy
    └── renderer (natural / flattened / inverted glyph frequencies)
            line crops (height 70 px) → manifests
                    train.py (fp16) → checkpoints
                            probes → data/probe_results/*.jsonl
                                    analysis → docs/ + paper figures
```

| Stage | Role |
|---|---|
| Scoring | `src/eval/equivalence_tables.py`, `transliteration_equivalence.py` |
| Data | `src/renderer/`, `src/data_pipeline/export_manifest_scaled.py` |
| Model | `src/models/instrument/` — ViT encoder, grapheme-cluster decoder, `train.py` |
| Probes | `src/probes/` — blank/noise, calibration, ablation, GT-likelihood, held-out transfer |
| Analysis | `src/analysis/` — statistics and figures |

Training: 5,000 steps, constant learning rate, last checkpoint, three seeds, vocabulary size ≈ 367. Flattened and inverted frequency conditions did not yield a usable reader (~0% line accuracy); mechanistic results use the three **natural** checkpoints.

---

## Probes

| Probe | Question |
|---|---|
| Blank / noise | Does confidence drop when there is nothing to read? |
| Training curve | Does a text-bearing vs blank gap appear as loss falls? |
| Calibration | Does confidence rank correctness on synthetic lines? |
| Zero-shot scripts | Same measurements on Ol Chiki and Perso-Arabic? |
| Encoder ablation | Does zeroing visual memory move the confidence peak? |
| GT-likelihood | Teacher-forced log *p*(GT) by position vs *n*-gram references |
| Held-out transfer | Does the pattern hold on held-out GlotOCR pages? |

Implementations: `src/probes/`. Per-probe write-ups: `docs/`. Interactive figures: the [project page](https://adya6714.github.io/vlm-ocr-eval/).

---

## Production OCR confidence

How this work relates to **Sarvam Vision** and Doc-AI Extract (published language accuracies vs Extract confidence, and why the from-scratch probes exist): [`docs/sarvam_vision_confidence.md`](docs/sarvam_vision_confidence.md). That evaluation is not part of the preprint.

---

## Repository layout

```
paper/                   Preprint (LaTeX + PDF)
index.html               Project page
src/eval/                Engines, encoding metrics, API client
src/renderer/            Controlled line rendering
src/data_pipeline/       Fetch and manifests
src/models/instrument/   Encoder, decoder, training, generation
src/probes/              Diagnostic probes
src/analysis/            Statistics and figures
data/manifests/          Training line lists
data/probe_results/      Probe outputs (committed)
docs/                    Analyses and measurement logs
```

Weights, raw page images, engine dumps, and API caches are gitignored. Training and OCR batches are intended for a single GPU (e.g. Colab T4); see `COLAB_RUNS.md`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make smoke-test    # architecture only; no findings
pytest -q
```

Rebuild paper figures from committed jsonl (no GPU):

```bash
PYTHONPATH=src/eval python3 src/analysis/make_paper_figures.py \
  --results-root data/probe_results \
  --out-dir docs/figures --paper-dir paper/figures
```

Compile the preprint:

```bash
tectonic -X compile paper/main.tex
```

Train (GPU recommended; checkpoints are not in git):

```bash
python src/models/instrument/train.py \
  --manifest data/manifests/hindi_natural.jsonl \
  --script hindi --condition natural --seed 0 \
  --output-root checkpoints
```

Further documentation: [`BOOK.md`](BOOK.md) (full project reference: questions, pipeline, findings, decisions, status), [`IMPLEMENTATION.md`](IMPLEMENTATION.md) (module checkboxes), [`docs/training_config.md`](docs/training_config.md) (hyperparameters).

---

## Follow-up measurements

Several items that used to sit here are **in the preprint**: noise /
patch-scrambled teacher-forcing, n-gram KL vs 5-gram, Probe 5 overlap
(empty held-out arm), PaddleOCR feasibility stop.

Still **inference-only** and still blocked on checkpoints in this
checkout: position-0 rank (jsonl not written), shuffled image–text
pairing, cross-attention contribution norms. A viable autoregressive
positive control is still open (Surya and PaddleOCR failed for
architecture).

Status: [`docs/remaining_measurements.md`](docs/remaining_measurements.md).
A larger demo model (LoRA SFT ran; RLVR policy did not —
[`docs/rlvr_scoping.md`](docs/rlvr_scoping.md)) is out of scope for the
current preprint.

---

## Citation

```bibtex
@misc{srivastava2026reading,
  title  = {Reading Without Looking: A Position-Resolved Profile of
            Confidence and Correctness in a Small Document OCR Model},
  author = {Srivastava, Adya},
  year   = {2026},
  url    = {https://github.com/Adya6714/vlm-ocr-eval}
}
```

Raw GlotOCR pages are not redistributed here; obtain them from the upstream benchmark. This repository is not an OCR leaderboard.

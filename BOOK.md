# BOOK.md

This file is the **single project reference**: the research questions,
how every result was produced, what the numbers mean, which stages are
built, and why the non-obvious design choices won.

> **This was not primarily an OCR-model-building project.** It was a
> controlled scientific investigation into whether an OCR model's
> confidence actually reflects **visual evidence** — or whether the
> decoder can be highly confident simply because the generated text
> looks linguistically plausible.
>
> **Critical framing:** The paper does **not** prove that a model that
> can read is hallucinating. It shows that, in this particular
> from-scratch instrument, confidence remains extremely high even when
> the model demonstrably does **not** read the evaluation image. Never
> claim "VLMs don't look at images" or "we proved OCR hallucinates."
>
> The ~19.6M-parameter model in the probes is a **measurement
> instrument**, not production OCR (19,607,104 params at |V|≈367;
> Decision #65). That distinction matters enormously when presenting
> this work to Sarvam or in a research interview.

Companion files still exist and stay authoritative for their jobs:

| File | Job |
|---|---|
| This `BOOK.md` | Narrative + pipeline + findings + decision *summaries* + status |
| [`AGENTS.md`](./AGENTS.md) | Workflow rules for agents (read order, when to flip checkboxes, Colab/resume requirements) |
| [`DECISIONS.md`](./DECISIONS.md) | Append-only log: Decision / Alternatives / Why. Numbered 1–89+. Do not rewrite past entries. |
| [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) | Module checklist, inputs/outputs, acceptance criteria, `[x]` / `[~]` / `[!]` |
| [`TODO.md`](./TODO.md) | Sequencing and pace; not the spec |
| [`docs/RESULTS.md`](./docs/RESULTS.md) | Index: claim → jsonl → analysis script |
| [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md) | Regenerable tables. **Cite this file; do not copy tables into chat or invent new headlines.** |
| [`docs/training_config.md`](./docs/training_config.md) | Hyperparameters and eval-set provenance |
| [`COLAB_RUNS.md`](./COLAB_RUNS.md) | What ran where (Colab vs laptop) and where artifacts landed |
| [`paper/main.pdf`](./paper/main.pdf) | Preprint (*Reading Without Looking*) |
| [`README.md`](./README.md) | Dual-audience map, not a second paper (Decision #76) |

If a number in this book disagrees with `docs/paper_defensibility_stats.md`
or a committed jsonl under `data/probe_results/`, the doc and the jsonl
win. Treat BOOK as the explanation of those files, not a second copy of
the tables.

---

## Preface

This book has two jobs at once.

**Teach.** Someone who has never opened this repo — and does not already
know computer vision or Indic scripts — should be able to read a chapter
and understand both the idea and why this project needed it.

**Record.** Someone who *has* been in the repo should be able to find,
without hunting through chat history: what we asked, how we built the
apparatus, which scripts produced which artifacts, what the probes
actually showed, which stages were deferred on purpose, and which
decision IDs lock those choices.

The intellectual center is **Question B** (below): does decoder
confidence track whether the image supports the text, or only whether
the output looks like language? The ~19.6M from-scratch model exists
so we can inspect full softmaxes, ablate the encoder, and run
position-resolved teacher forcing — things a closed API cannot do. It
is deliberately **not** a production OCR engine.

**Why Sarvam cares.** Production Indic OCR (including Sarvam Vision)
exposes confidence for routing and review, yet published per-language
accuracy varies widely (Hindi **95.91%**, Santhali **80.32%**,
Kashmiri **55.93%** word accuracy on Sarvam's Indic OCR benchmark —
re-verified on [sarvam.ai/blogs/sarvam-vision](https://www.sarvam.ai/blogs/sarvam-vision),
Decisions #6, #60). Stage 5 is **not** an accuracy competition against
Sarvam; it asks whether confidence signals track difficulty. See
**Presenting to Sarvam** (before Chapter 0) for the pitch.

Read **First principles: the central finding** before the teaching
chapters if you want the mechanistic story step-by-step without code.

**Preprint vs later extensions.** [`paper/main.pdf`](./paper/main.pdf)
covers Stage 0–2 and the mechanistic probes (Stage 4). **Stage 5b**
(rank-correlation transfer) and **Stage 6** (offline triage cascade) were
computed and documented afterward in `docs/stage5b_rank_correlation.md`
and `docs/stage6_triage_cascade.md`; they are **not** in the preprint.
When presenting the work, say: *the preprint focuses on the instrument;
we subsequently extended with production-confidence transfer and offline
triage analysis.*

**Executed in this phase:** Stage 0 (error taxonomy), Stage 1 (controlled
renderer + line manifests), Stage 2a (from-scratch instrument), Stage 4
probes, Stage 5a (Sarvam Extract confidence vs published accuracy),
Stage 5b (rank-correlation transfer, offline), Stage 6 (triage cascade,
offline).

**Engineering that makes the science runnable:** resume-by-default
baselines, hand-review suggestions, line-crop export, `make smoke-test`,
Colab `--data-root` + zip export.

**Specified, taught, not executed:** RLVR **policy** training (coverage
ablation still untrained; `docs/rlvr_scoping.md`); Stage 2b **SFT** ran
on Colab T4 (Decision #84). Noise/scramble, n-gram KL, and the Probe 5
overlap check are in the preprint.

Numbers recomputed from this checkout are labeled **measured**. Numbers
that appear on the site or in older write-ups but whose result files are
not in the tree are labeled **reported**.

---

## How to read this book

| If you want… | Start here |
|---|---|
| The central finding from first principles | **First principles: the central finding** (before Chapter 0) |
| What ran vs what we did not prove vs next steps | **Experiments done · Not proved · Roadmap** (after **Presenting to Sarvam**) |
| A Sarvam call or product presentation | **Presenting to Sarvam** + **60-second story** |
| The 60-second interview pitch | **60-second story** (below) |
| Defending the project in an interview | **Interview defense** (below) + **Appendix H — Interview Q&A** + **What not to claim** |
| How this relates to published VLM/OCR work | **Related work** (before Chapter 0) |
| A quick repo map + what runs where | **Start here (15 minutes)** |
| The research chain in one diagram | **The entire project in one picture** |
| The core conceptual distinction | **Correctness ≠ confidence ≠ grounding** · **What the paper does and does not claim** |
| The research questions and what we concluded | **Research questions and answers** · **What we concluded** |
| How data flowed from GlotOCR pages to paper figures | **How the results were built** |
| What exists vs what is still a checkbox | **Implementation status** |
| Why a weird-looking choice exists | **Decision catalog** or Appendix A → `DECISIONS.md` |
| Computer vision / Indic scripts from zero | Chapter 0 (after **First principles**) |
| Why Indic CER is a measurement problem | Chapter 1 |
| The renderer and exposure dial | Chapter 2 |
| The instrument architecture and training | Chapter 3 |
| The probe findings in detail | Chapter 7 |
| Production API vs owned model | Chapter 8 |
| Interview attack questions | **Appendix H — Interview Q&A** |
| Commands to regenerate headlines | Appendix E |

Teaching chapters keep this shape:

1. The question that forced the chapter.
2. The concept in plain language.
3. The design choice — alternatives and why this won (decision IDs).
4. What got built, with paths.
5. What the evidence shows, or what is still missing.
6. One sentence to keep.

### 60-second story

> Modern Indic OCR can achieve high aggregate accuracy while performance
> varies substantially across scripts. Sarvam's published benchmark shows
> Hindi **95.91%**, Santhali **80.32%**, and Kashmiri **55.93%** word
> accuracy — a **39.98 pp** spread — yet production systems still expose
> a confidence scalar for routing. The practical question is not only
> whether OCR is wrong, but whether confidence tells us **when** it is
> wrong and **why**.
>
> I could not answer that with a production API alone — I needed
> counterfactuals: blank images, unseen scripts, encoder ablation, and
> full token distributions. So I built a ~19.6M-parameter encoder-decoder
> **from scratch** as a measurement instrument (not a competitor to
> Sarvam), plus Indic-aware grapheme scoring and a controlled renderer.
>
> The sharpest finding: at **generation position 0** (no text prefix yet),
> the instrument assigns teacher-forced *p*(correct first grapheme) ≈
> **2.2×10⁻¹¹** while self-generated max-softmax ≈ **0.90**; ground truth
> is never the argmax (**0/180**). Blank images behave similarly. This
> does **not** prove that models which can read are hallucinating — it
> shows confidence without visual grounding in **this** undertrained
> instrument. Mid-sequence behavior matches a 4–5-gram text-only LM
> (KL ≈ **0.27–0.32** nats; [`docs/ngram_kl_argmax.md`](./docs/ngram_kl_argmax.md)).
>
> External validation on Sarvam Extract: page-level confidence moved only
> **0.0027** across the published language gap (Stage 5a); per-image
> difficulty did not predict Sarvam CER (Stage 5b: ρ = **0.0293**,
> p = **0.8267**); instrument-confidence routing was worse than random at
> 20% escalation (Stage 6). The lesson for production: **max-softmax is
> not the same as visual grounding** — the next step is a
> grounding-aware confidence signal.

Numbers: [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md).
Do not copy tables into slides from memory.

### Interview defense

Use this spine when someone asks you to walk through the methodology.
Each step was designed to eliminate a competing explanation — not merely
to accumulate plots.

1. **Measurement** — grapheme-cluster alignment and Tier-1 encoding
   equivalence before counting errors (Stage 0).
2. **Controlled data** — renderer with natural / flattened / inverted
   exposure; TV dial verified (natural **0**, flat ≈ **0.047**, inv ≈
   **0.005** — [`docs/RESULTS.md`](./docs/RESULTS.md)).
3. **Controlled model** — from-scratch encoder-decoder so pretraining
   cannot confound exposure (Stage 2a; 19,607,104 params —
   [`docs/training_config.md`](./docs/training_config.md)).
4. **Instrumentation** — full softmax, teacher forcing, encoder ablation.
5. **Behavioral tests** — text-bearing vs blank vs noise vs unseen scripts
   (Probe 3, 5b).
6. **Mechanistic test** — zero encoder memory: pooled mean confidence
   full **0.9861** vs zero **0.9891** (Δ **−0.0030**); mean KL **1.075**
   nats; top-1 agree ≈ **0.879**; prior sufficiency **0.8827**
   ([`docs/attention_ablation_analysis.md`](./docs/attention_ablation_analysis.md)).
7. **Position-resolved test** — position 0 isolates vision; mid-sequence
   ≈ 4/5-gram LM (position-0 instrument **−24.54** nats vs n-gram first
   symbol **−4.73** — [`docs/position_matched_ngrams.md`](./docs/position_matched_ngrams.md)).
8. **Exposure experiment (Probe 1)** — dial worked; causal β withheld
   (natural **18.3%** vs flattened **0.33%** / inverted **0.67%** line
   accuracy — [`docs/probe1_fixed_effects.md`](./docs/probe1_fixed_effects.md)).
9. **External validation** — Sarvam Extract vs published language spread
   (Stage 5a); per-image rank transfer (Stage 5b: null ρ).
10. **Operational test** — triage cascade (Stage 6: instrument worse than
    random at k=12 / 20%).
11. **Limitation** — no positive control (known-good AR Devanagari reader
    under the same position-0 diagnostic); no genuine held-out synthetic
    train/eval split before training; AUROC **0.838** is in-manifest only
    (held-out-image AUROC(conf→CER<median) ≈ **0.5698** —
    [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)).

**If they say "your model barely reads":** accuracy was not the objective;
the instrument exists so we can inspect distributions and run
counterfactuals a closed API cannot. See **Appendix H**.

**If they say "why not just Sarvam":** Sarvam is the external production
reference; the instrument is the mechanistic lab. Stage 5 tests transfer,
not numerical comparability of confidence semantics.

---

## The entire project in one picture

```text
PROBLEM — How trustworthy is OCR confidence?
   ↓
Stage 0 — Indic-aware evaluation (what counts as "wrong"?)
   ↓
Stage 1 — Controlled renderer (control what the model sees)
   ↓
Stage 2 — From-scratch OCR instrument (known empty history)
   ↓
Probes — exposure · confusion · blank/noise · calibration ·
         unseen scripts · encoder ablation · teacher-forced p(GT) by position
   ↓
Production transfer — Does this rhyme with Sarvam confidence?
   ↓
Triage — Can confidence route pages to a stronger system?
```

**Central discovery:** in this from-scratch instrument, the model can be
extremely confident even when it demonstrably does not read the
evaluation image — **not** a universal claim that models which can read
are hallucinating. The sharpest case is **generation position 0** (no
prefix yet — only the image can supply input-dependent information):

- self-generated max-softmax ≈ **0.90**
- teacher-forced *p*(correct first grapheme) ≈ **2.2 × 10⁻¹¹**
- ground truth is the argmax in **0 / 180** sequences
- text-bearing and blank behave similarly

Numbers: [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md).
Do not treat chat summaries as canonical.

---

## Correctness ≠ confidence ≠ grounding

```text
CORRECTNESS — Did the model output the right text?     → often LOW
GROUNDING   — Did the image support that output?       → LOW
CONFIDENCE  — How peaked is the output distribution?   → HIGH
```

Max-softmax only answers the third. You might assume confidence ≈
grounding ≈ correctness; this project shows they can separate. That is
why the preprint is called *Reading Without Looking*.

### What the paper does and does not claim

| Does claim (in this instrument + sample) | Does **not** claim |
|---|---|
| Max-softmax can stay near ceiling when *p*(GT) at position 0 is ~10⁻¹¹ and GT is never argmax (0/180) | That VLMs or production OCR "don't look at images" in general |
| Blank, unseen-script, and ablation controls are consistent with a language-prior-dominated confidence peak | That Sarvam's encoder can be blank-ablated or that their confidence is "broken" |
| Mid-sequence teacher-forced behavior matches a 4–5-gram text-only LM (KL ~0.27–0.32 nats mid-sequence) | That exposure **causes** better OCR (Probe 1 causal β withheld) |
| Sarvam Extract page confidence moved **0.0027** across a **39.98 pp** published accuracy gap (Stage 5a, n=35) | That our 20M instrument should match Sarvam accuracy |
| Per-image instrument difficulty did not predict Sarvam CER (ρ = **0.0293**, p = **0.8267**, Stage 5b) | That AUROC **0.838** generalizes (in-training-manifest only) |
| Instrument-confidence routing left higher residual CER than random at 20% escalation (Stage 6) | That max-softmax is useless everywhere — blanks scored 0.0 on Sarvam Extract |

> **Repeat this framing early and often:** the paper shows **confidence
> without visual grounding in a controlled from-scratch instrument**, then
> tests weak/external transfer to production — not universal hallucination.

---

## Start here (15 minutes)

If you are reading this repo for the first time, the fastest way to get
oriented is to separate three things:

- **The paper narrative** (`paper/main.pdf`) — what we claim and why
  (instrument + mechanistic probes only).
- **The committed evidence** (`data/probe_results/*.jsonl`, plus
  `docs/*_analysis.md`) — what you can regenerate from this checkout.
- **The heavy compute** (Colab checkpoints, `data/raw/*`, `data/cache/*`)
  — what is intentionally *not* in git.

The ~19.6M model is a **measurement instrument** for probing decoder
confidence — not something you deploy for OCR accuracy. If you plan to
modify the repo, read `AGENTS.md` first — it defines the stage order,
what "done" means, and the Colab/resume rules this project treats as
non-negotiable.

### Glossary (terms used everywhere)

- **Instrument vs demo**: the "instrument" is the from-scratch model used
  for probes (it has deliberately empty pretraining exposure); the "demo"
  is the LoRA-adapted VLM whose job is to resemble a production system's
  components (Decision #1). The scientific confidence question is
  answered by the instrument, not the demo.
- **Tier 0/1/2 (scoring)**: Tier 0 = whitespace normalization; Tier 1 =
  deterministic encoding equivalence; Tier 2 = phonetic equivalence via
  transliteration (Decisions #4, #7, #8).
- **Tier A/B/C (data realism)**: Tier A = clean controlled renders; Tier
  B = degraded controlled renders; Tier C = held-out GlotOCR evaluation
  images (Decision #74: they are **renders**, not photographs or scans).
- **Teacher forcing**: score \(p(\mathrm{GT})\) on the ground-truth next
  token under the ground-truth prefix, instead of letting the model follow
  its own argmax path.
- **Position 0**: the first generated token. There is no prefix yet, so
  any high probability on the correct first grapheme must come from the
  image, not a language prior.

### What you can run on a laptop (no GPU)

- **Architecture proof (fake data, no `data/raw/`)**: `make smoke-test`  
  This is the "does the pipeline wire together?" check. It does not
  produce findings.
- **Recompute offline paper tables from committed jsonl**:
  `python3 src/analysis/paper_defensibility_stats.py`  
  Writes/overwrites `docs/paper_defensibility_stats.md`.
- **Regenerate figures from committed jsonl**:
  `python3 src/analysis/make_paper_figures.py`  
  Writes `paper/figures/*.pdf` and `docs/figures/*.png` (see Decision
  #64).
- **Stage 5b / Stage 6 reports** (offline, no API): already in
  `docs/stage5b_rank_correlation.md` and `docs/stage6_triage_cascade.md`.

### What requires Colab T4 (checkpoints live off-repo)

- **Training the instrument**: `src/models/instrument/train.py` (fp16 on
  free Colab T4; resumable by default — see hard constraints in
  `AGENTS.md`)
- **Probes that forward-pass the trained instrument**: most of
  `src/probes/`  
  These write committed JSONL outputs under `data/probe_results/`.

`COLAB_RUNS.md` is the short "what ran where" ledger.

### Repo structure (what lives where)

This is the map most beginners want before reading any chapter:

```
src/
  data_pipeline/        fetch + export (GlotOCR fetch, line-crop manifests)
  eval/                 Stage 0: baselines + Tier 0/1/2 scoring + taxonomy + hand review
  renderer/             Stage 1: layout bank + degradation + HarfBuzz render + glyph-frequency dial
  models/
    instrument/         Stage 2a: from-scratch OCR model used for probes (the "instrument")
    demo/               Stage 2b: LoRA + SFT + RLVR reward scaffolding (the "demo")
  probes/               Stage 4/5: runs that emit `data/probe_results/*.jsonl`
  analysis/             scripts that turn jsonl into `docs/*_analysis.md` + figures

data/
  probe_results/        committed JSONL evidence (small; must be pushed after Colab)
  manifests/            committed line-crop manifests (training inputs)
  raw/                  GlotOCR images + GT (gitignored; fetch upstream)
  cache/                caches (gitignored; incl. Sarvam JSON responses)

docs/                   regenerable reports (source-of-truth tables live here)
paper/                  LaTeX + publication figures + compiled PDF
```

### Pipeline at a glance

Two pictures: one for **data flow**, one for **what to touch** when you
want a new result.

```mermaid
flowchart LR
  RAW[data/raw/* (gitignored)] -->|export manifests| MAN[data/manifests/*.jsonl (committed)]
  MAN -->|train on Colab T4| CKPT[checkpoints/*.pt (not in git)]
  CKPT -->|run probes| PR[data/probe_results/*.jsonl (committed)]
  PR -->|analysis scripts| DOCS[docs/*_analysis.md (committed)]
  DOCS --> FIGS[paper/figures/*.pdf + docs/figures/*.png]
```

```mermaid
flowchart TD
  A[You want a new number/plot] --> B{Where should it live?}
  B -->|Raw model outputs| PR[data/probe_results/*.jsonl]
  B -->|Aggregated narrative + plots| DOCS[docs/*_analysis.md]
  B -->|Paper-ready figure| FIGS[paper/figures/*.pdf]
  B -->|Design rationale| DEC[DECISIONS.md entry]
  B -->|Build status| IMPL[IMPLEMENTATION.md checkbox]
  PR --> DOCS
  DOCS --> FIGS
```

### Where to find "the results" (links, not duplicated tables)

This book explains. These files *are the evidence*:

- **Canonical result index**: [`docs/RESULTS.md`](./docs/RESULTS.md)
- **Regenerable paper tables**:
  [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)
- **Stage 0 measurement + validation**:
  - [`docs/tier0e_paddleocr.md`](./docs/tier0e_paddleocr.md) (PaddleOCR
    corpus fill + taxonomy slice)
  - [`docs/adjudication_analysis.md`](./docs/adjudication_analysis.md)
    (UNREVIEWED sample + what's still provisional)
  - [`docs/tier2_validation.md`](./docs/tier2_validation.md) (Tier 2
    hand-checked pairs)
- **Probe write-ups**:
  - [`docs/probe5b_analysis.md`](./docs/probe5b_analysis.md) (zero-shot
    floor)
  - [`docs/attention_ablation_analysis.md`](./docs/attention_ablation_analysis.md)
    (encoder-memory ablation)
  - [`docs/gt_likelihood_analysis.md`](./docs/gt_likelihood_analysis.md)
    (teacher-forced log \(p(\mathrm{GT})\) + entropy)
  - [`docs/probe6_synthetic_real_analysis.md`](./docs/probe6_synthetic_real_analysis.md)
    (synthetic → held-out Tier C check)
- **Production API audit**:
  [`docs/sarvam_vision_confidence.md`](./docs/sarvam_vision_confidence.md)
- **Stage 5b / Stage 6 (post-preprint, offline)**:
  [`docs/stage5b_rank_correlation.md`](./docs/stage5b_rank_correlation.md),
  [`docs/stage6_triage_cascade.md`](./docs/stage6_triage_cascade.md)
- **Where runs happened**: [`COLAB_RUNS.md`](./COLAB_RUNS.md)
- **The paper narrative**: [`paper/main.pdf`](./paper/main.pdf)

### "Jump to evidence": each research question → the backing artifact

If you only have time for the defensible core, this is the fastest path.

- **Q A (exposure vs complexity)**: Probe 1 fixed-effects diagnostic  
  - Script: `src/analysis/probe1_fixed_effects.py`  
  - Report: `docs/probe1_fixed_effects.md`
- **Q B (confidence = visual evidence?)**: blank/noise + training curve
  + ablations + teacher forcing + position 0  
  - Probe 3: `data/probe_results/probe3_hindi_*.jsonl`  
  - Probe 3b: `data/probe_results/probe3_curve_*.json` +
    `docs/probe3_curve_analysis.md`  
  - Attention ablation: `data/probe_results/attention_ablation_*.jsonl` +
    [`docs/attention_ablation_analysis.md`](./docs/attention_ablation_analysis.md)  
  - Teacher-forced likelihood:
    `data/probe_results/probe_gt_likelihood_*.jsonl` +
    [`docs/gt_likelihood_analysis.md`](./docs/gt_likelihood_analysis.md)  
  - Headline numbers:
    [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)
    (Abstract Fact + Items 1–7)
- **Q1 (what counts as an error?)**: Stage 0 scoring + taxonomy  
  - Code: `src/eval/error_taxonomy.py`, `equivalence_tables.py`,
    `transliteration_equivalence.py`  
  - Report: [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)
    (Tier 1 fraction; see also
    [`docs/adjudication_analysis.md`](./docs/adjudication_analysis.md))
- **Q2 (exposure vs complexity)**: same as Q A above
- **Q3 (reading vs guessing)**: same as Q B above
- **Q4 (softmax rank of true grapheme)**: Probe 2  
  - JSONL: `data/probe_results/probe2_hindi_natural_seed*.jsonl`  
  - Report: `docs/probe2_confusion_analysis.md`
- **Q5 (calibration)**: Probe 5 + offline defensibility battery  
  - JSONL: `data/probe_results/probe5_hindi_*.jsonl`  
  - Tables: [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)
- **Q6 (synthetic → Tier C check)**: Probe 6  
  - JSONL: `data/probe_results/probe6_synthetic_real_hindi_seed*.jsonl`  
  - Report: [`docs/probe6_synthetic_real_analysis.md`](./docs/probe6_synthetic_real_analysis.md)
- **Q7 (production confidence vs published accuracy)**: Stage 5a + 5b  
  - JSONL: `data/probe_results/sarvam_transfer_probe.jsonl`  
  - Reports: [`docs/sarvam_vision_confidence.md`](./docs/sarvam_vision_confidence.md),
    [`docs/stage5b_rank_correlation.md`](./docs/stage5b_rank_correlation.md)
- **Q8 (triage cascade)**: Stage 6 (offline)  
  - Report: [`docs/stage6_triage_cascade.md`](./docs/stage6_triage_cascade.md)  
  - Code: `src/probes/cascade.py`

---

## Research questions and answers

The project is a controlled scientific investigation of **decoder
confidence** in Indic document OCR — not primarily an attempt to build
the most accurate OCR model. Production systems use per-token or
per-page confidence to decide what a human should review. The central
question is whether that signal tracks **whether the image supports the
text**, or only **whether the generated string looks like language**.

### Question A — Why is Indic OCR difficult?

Is poor recognition caused by **insufficient grapheme exposure**, or by
**intrinsic visual complexity**? Natural Devanagari frequency spans
orders of magnitude; accuracy-vs-frequency confounds exposure with
complexity. That motivated the controlled renderer and the natural /
flattened / inverted exposure conditions (Stage 1, Probe 1). See **Q2**
below for what we found — including the distinction that **apparatus
success ≠ experiment success** when flattened/inverted models sit at the
performance floor.

### Question B — Does confidence mean the model actually sees the image?

This is the **central paper question**. Production OCR often assumes:
high confidence → the model knows; low confidence → send to a human.
But an autoregressive decoder can emit fluent text from a language prior
while the image is blank, scrambled, or an unseen script. The informative
test is **position 0**: there is no generated prefix, so any probability
on the correct first grapheme has to come from the image.

**Position-0 headline** (cite
[`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md),
Abstract Fact):

| Signal | Text-bearing Hindi | Blank |
|---|---:|---:|
| Self-generated max-softmax (pos 1) | **0.9020** | **0.8952** |
| Teacher-forced *p*(GT) at pos 0 | **2.204×10⁻¹¹** | **6.328×10⁻¹¹** |
| GT is argmax at pos 0 | **0 / 180** | **0 / 180** |

The model is ~90% confident in its chosen first token while assigning
the true first grapheme ~10⁻¹¹ probability. See **Q3–Q6** for the full
probe battery.

Indic scripts add two measurement problems before either question is
answerable: multiple valid Unicode encodings of the same reading, and
published systems that fail **non-silently** — high confidence when the
script in the image is one the model cannot read. A closed API cannot
explain *why*. This repository therefore (1) normalizes encodings before
counting residual errors, (2) trains a small reader with a known empty
Indic pretraining history, and (3) probes blank input, unseen scripts,
encoder ablation, and teacher-forced likelihood **by generation position**.

### Q1 — When an engine disagrees with ground truth, is it wrong?

**Question.** Exact-match and code-point CER treat every byte
disagreement as an error. Indic orthography has encoding variants NFC
does not cover (ZWJ/ZWNJ, anusvara vs conjunct nasal, khanda-ta, danda
vs period, digit systems). How much of "error" is spelling, not
misreading?

**How we asked it.** Stage 0: run Tesseract / Surya / PaddleOCR
(`src/eval/run_baselines.py`), align at grapheme-cluster level
(Decision #7), apply Tier 0 whitespace, Tier 1 encoding tables
(`equivalence_tables.py`, Decision #4: NFC is already upstream), Tier 2
ISO 15919 phonetic match (`transliteration_equivalence.py`, Decisions
#8, #18, #54), then human residual labels (`hand_review.py`).

**What we found.** On Tesseract, **measured** ~20.4% of *non-exact*
rows are Tier 1 — same reading, different bytes. That headline stays
**provisional** until the n=200 adjudication sample is labeled and the
bootstrap CI is reported (Decision #55, `docs/adjudication_analysis.md`).
Corpus Tier 2 after Tier 1 is **0%** on the current predictions — phonetic
residuals are rare here, which is itself a finding (#54). UNREVIEWED
is still large. Probe 4 (same scorers on *instrument* output) is not
re-run as a separate campaign.

**Implied fix.** Score Indic OCR grapheme-aware and Tier-1-aware before
comparing engines or claiming improvement.

### Q2 — Is a glyph hard because it is visually hard, or because it was rare?

**Question.** Same as Question A. Natural Devanagari frequency spans
orders of magnitude. Accuracy vs frequency confounds exposure with
complexity.

**How we asked it.** Stage 1 renderer with a glyph-frequency *dial*:
`natural` / `flattened` / `inverted`, same data volume, TV ≤ 0.08
(Decisions #10, #25, #29). Train the instrument nine times (3
conditions × 3 seeds, Decision #14). Fit per-cluster accuracy on log
exposure with glyph fixed effects (`probe1_fixed_effects.py`).

**What we found.** The dial moved. Flattened/inverted **collapsed line
accuracy to ~0%**; natural reached only modest line accuracy (~18% in
the first FE diagnostic). Headline β is **withheld** as uninterpretable
(Decision #49, `docs/probe1_fixed_effects.md`). **Apparatus success ≠
experiment success:** we demonstrated exposure manipulation, but the
planned causal slope is not claimable because two conditions never
learned to read. Mechanistic paper claims therefore use the three
**natural** checkpoints only (#66), not all nine.

**Implied fix.** Treat exposure as something you design. Do not report
an exposure slope when two of three conditions never learned to read.

### Q3 — Is the model looking at the image, or completing language?

**Question.** Same as Question B. Autoregressive OCR can emit fluent
text from a language prior. Does confidence drop when there is nothing
to read? Does it appear as training loss falls? Does zeroing encoder
memory change the confidence peak? Does teacher-forced *p*(ground truth)
at position 0 exceed chance?

**How we asked it.** Probe 3 blank/noise; Probe 3b snapshots at steps
500/1000/2000/3000/5000; attention ablation (zero encoder memory before
`memory_projection`, Decision #56); GT-likelihood teacher forcing
(Decision #62); Probe 5b Ol Chiki and Perso-Arabic (true zero script
exposure, Decisions #6, #50–#53).

**What we found.** Confidence stays near ceiling on text-bearing Hindi,
blank, noise, Santhali, and Kashmiri (condition means within a few
thousandths; 5b between-condition range of across-seed means **0.0037**,
smaller than seed SDs). Training-curve real−blank gap sign-flips and is
indistinguishable from zero across seeds. Ablation: mean confidence
almost unchanged (full vs zero-memory); ~8% of token-choice mass
depends on image content. Teacher-forced first-token *p*(GT) is on the
order of 10⁻¹¹ while self-generated max-softmax is ~0.90. Ground truth
is never the argmax at position 0 (0/180 sequences). Charset: **360/360**
unseen-script images emitted **zero** characters of the image script.
CER on held-out Hindi is near 1 and not better than blank (see
`docs/paper_defensibility_stats.md` Item 1). The model **is**
undertrained; the finding is that confidence **does not report that**.

**Implied fix.** Do not use max-softmax as a "the image supports this
string" signal on this class of decoder. Position 0 and teacher-forced
*p*(GT) are the honest tests.

### Q4 — When the model is wrong, does the softmax still know the right cluster?

**Question.** Closed APIs return one string. If *p*(true) sits at rank
2–3, the encoder might carry signal that argmax fails to surface.

**How we asked it.** Probe 2: GT-aligned substitutions plus *p*(true)
and rank from the full softmax (Decision #57).

**What we found.** See `docs/probe2_confusion_analysis.md` (jsonl
`probe2_hindi_natural_seed{0,1,2}.jsonl`). Mixed top pairs; substantial
EOS/space mass. This is not a story of "almost right, rank 2."

### Q5 — Does confidence rank correctness?

**Question.** Calibration: among predictions at confidence *c*, is
accuracy ≈ *c*? Does calibration break on starved glyphs (Probe 1 × 5)?

**How we asked it.** `probe5_calibration.py` on all nine Hindi runs;
offline equal-mass ECE in `paper_defensibility_stats.py` (Decision #64).

**What we found.** Natural: confidence ~99%, line accuracy in the
mid-teens, ECE ≈ 0.81. Flattened/inverted: accuracy ~0–1%. Cite
`docs/paper_defensibility_stats.md` for the pooled AUROC. That AUROC
is **not** a held-out-string result: Probe 5 samples its eval rows from
`hindi_natural.jsonl` itself, so the non-overlapping subset is empty
(`docs/memorisation_split.md`). A different pool (60 teacher-forced
real-scan strings) has partial overlap with the same manifest; that is
not the Probe 5 split. Mid-sequence teacher-forced log *p*(GT) sits in
the same band as a 4–5-gram grapheme LM
(`docs/position_matched_ngrams.md`). Full-softmax KL vs that prior
**is computed** on three seeds (`docs/ngram_kl_argmax.md`): mid-sequence
KL is ~0.27–0.32 nats with ~91–94% argmax agreement; positions 0, 1,
and 40+ diverge. The paper treats the exclusive 5-gram account as
supported.

**Implied fix.** Report calibration with equal-mass bins and ECE; never
treat AUROC on overlapping strings as generalization.

### Q6 — Does the synthetic world lie? Does the pattern hold on held-out pages?

**Question.** Controlled renders (Tier A/B) might not match evaluation
pages (Tier C). Also: train/eval leakage.

**How we asked it.** Probe 6 paper scope (Decision #58): Tier C Hindi
plain + degraded + blank vs synthetic Claim B. Leakage check: 0 overlaps
between `data/manifests/hindi_*.jsonl` and `data/raw/hindi/images/`.
Eval images are still GlotOCR **renders**, not photographs (Decision
#74: never label them "real" in figures).

**What we found.** Mean confidence stays high (plain / degraded / blank
in the same band); Tier 1/2 line accuracy 0.0 on the scored real
sample. Pattern transfers across that gap; reading does not appear.
Details: `docs/probe6_synthetic_real_analysis.md`.

### Q7 — Does a production API's confidence track its own published accuracy gap?

**Question.** The instrument is ~20M parameters and does not read.
Sarvam Vision is a production 3B-class system with published
per-language accuracies. Does Extract *confidence* fall when published
accuracy falls (Hindi → Kashmiri)?

**How we asked it.** Stage 5a for the language-gap probe. Extract only
(Digitise has no field confidence, Decision #59). 35 pages, cached once
(Decision #19). Published accuracies re-fetched from sarvam.ai
(Decisions #6, #60). Glyph-class transfer (#15) stays deferred; the
**per-image** Spearman transfer (Decision #89) is Stage 5b.

**What we found (5a).** Hindi → Kashmiri published accuracy gap
**39.98 pp**; Extract confidence delta **0.0027**. Blanks score 0.0000.
Write-up: `docs/sarvam_vision_confidence.md` (not in the preprint).

**What we found (5b).** Primary Hindi plains, Sarvam CER, n=60:
Spearman ρ = **0.0293**, permutation p = **0.8267** — a genuine null
(protocol locked before looking; Decision #89). Full table:
`docs/stage5b_rank_correlation.md`. See Chapter 8.

**What it does not prove.** That Sarvam "reads without looking" in the
instrument's mechanistic sense — we cannot blank-ablate their encoder.
It shows that **page-level confidence did not track the published
language-wise accuracy spread** in our sample, and that the instrument's
difficulty ranking did not predict Sarvam's per-image CER on the same
plains.

### Q8 — Can confidence route work to a stronger system or a human?

**Question.** Stage 6 cascade: sweep thresholds offline on the Stage 5
cache; compare to random / layout-complexity / Tesseract-confidence
escalation. Metric is router quality, not cost savings (Decision #16),
because the instrument is not expected to beat Tesseract.

**What we found.** At the pre-specified fair point (k=12, 20% of n=60
Hindi plains), instrument-confidence routing leaves residual system CER
**0.0109** — worse than random (**0.0095**), GT grapheme-length
(**0.0092**), and Tesseract confidence (**0.0079**). Report:
`docs/stage6_triage_cascade.md` (not in the preprint). On this test the
instrument confidence signal is **not** a useful production triage router.
See Chapter 9.

---

## What we concluded

Concise findings — cite the docs for tables; this list is orientation
only.

**Finding 1 — Indic evaluation itself can mislead.** ~20.4% of
Tesseract *non-exact* rows are Tier-1 encoding variants (provisional;
large UNREVIEWED). Measure at grapheme/encoding-aware level before
comparing engines.

**Finding 2 — Exposure apparatus worked; causal slope did not.**
Flattened/inverted training collapsed (~0% line accuracy); natural
~18%. Do not claim an exposure β from this experiment (Decision #49).

**Finding 3 — The instrument does not read held-out images; confidence
does not say so.** Grapheme CER ≈ 0.985 text-bearing vs ≈ 0.949 blank
(`docs/paper_defensibility_stats.md` Item 1); confidence ~99%.

**Finding 4 — Max-softmax is not a reliable grounding signal.** Blank,
unseen scripts, and encoder ablation do not meaningfully lower
confidence; position-0 *p*(GT) ~10⁻¹¹ vs max-softmax ~0.90.

**Finding 5 — Position 0 is the killer diagnostic.** No prefix → image
must matter; yet GT is never argmax (0/180) and self-confidence stays
~0.90.

**Finding 6 — Mid-sequence behaves like a text-only LM.** Teacher-forced
log *p*(GT) converges toward 4–5-gram grapheme priors; real and blank
curves overlap (KL ~0.27–0.32 nats mid-sequence).

**Finding 7 — Calibration is broken; AUROC is not held-out.** Natural:
conf ≈ 99.4%, acc ≈ 18.3%, ECE ≈ 0.811; AUROC 0.838 is
in-training-manifest only (`docs/memorisation_split.md`).

**Finding 8 — Production transfer: null rank correlation.** Instrument
difficulty ≉ Sarvam per-image CER: ρ = 0.0293, p = 0.8267 (Stage 5b).

**Finding 9 — Triage: worse than random.** Instrument-confidence routing
at 20% escalation leaves higher residual CER than random or Tesseract
confidence (Stage 6).

**Central limitation:** no positive control — a known-good autoregressive
Devanagari reader under the same position-0 diagnostic. Second: no
genuine held-out synthetic train/eval split before training.

---

## What not to claim

| Don't say | Say instead |
|---|---|
| "We proved OCR models hallucinate." / "VLMs don't look at images." | In **this** from-scratch instrument, confidence stayed high when the model demonstrably did not read the evaluation image; production evidence is weaker and correlational. |
| "We proved a model that can read is hallucinating." | We studied an undertrained instrument (~CER 0.985); the finding is **confidence ≠ grounding**, not universal VLM failure. |
| "We trained a production OCR model." | ~19.6M research instrument for mechanistic evaluation. |
| "18.3% held-out accuracy." | Synthetic line accuracy **18.3%** on the **training-manifest** regime (natural; [`docs/probe1_fixed_effects.md`](./docs/probe1_fixed_effects.md)). |
| "GlotOCR provided real scans." | Evaluation images are **renders** (plain / degraded). |
| "We beat / competed with Sarvam on accuracy." | Stage 5 is **not** an accuracy contest; it tests whether confidence tracks difficulty. |
| "We proved Sarvam doesn't look at images." | Page-level Extract confidence didn't track published language accuracy (Δ **0.0027**); instrument ranking didn't predict Sarvam CER (ρ **0.0293**). We cannot ablate their encoder. |
| "Sarvam confidence and our confidence are the same number." | Different semantics; Stage 5 tests two **transfer questions**, not calibration between models. |
| "AUROC 0.838 proves confidence generalizes." | AUROC is in-training-manifest only; held-out-image AUROC(conf→CER<median) ≈ **0.5698** ([`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)). |
| "Probe 1 proved exposure causes better OCR." | Apparatus manipulated exposure (TV passed); causal β withheld because flattened (**0.33%**) / inverted (**0.67%**) never learned to read. |
| "Present to Sarvam: our model vs theirs." | Present: instrument → mechanism → external validation → **grounding-aware confidence** as the engineering next step. |

---

## How the results were built

End-to-end data flow (same picture as `docs/RESULTS.md`):

```
data/raw/{hindi,bengali,santhali,kashmiri}/     GlotOCR images + GT (gitignored)
        │
        ├─► src/eval/run_baselines.py  →  data/predictions/{engine}/{language}.jsonl
        │         └─► error_taxonomy.py → error_taxonomy.csv
        │
        └─► src/renderer + export_manifest_scaled.py
                  └─► data/manifests/{script}_{mode}.jsonl     (committed)
                            └─► src/models/instrument/train.py   (Colab T4)
                                      └─► checkpoints/*.pt        (not in git)
                                                └─► src/probes/*.py
                                                          └─► data/probe_results/*.jsonl  (committed)
                                                                    └─► src/analysis/*.py
                                                                              └─► docs/*_analysis.md
                                                                              └─► paper/figures/*.pdf
```

### Data

- **Source.** GlotOCR-bench pages via `src/data_pipeline/fetch_glotocr.py`
  into `data/raw/{language}/` (plain + degraded PNGs +
  `ground_truth.jsonl`). Raw images are not redistributed; obtain them
  upstream.
- **Script scope (Decision #6).** Devanagari (Hindi) is the powered
  experiment. Bengali is a cross-script structural check (manifests
  exist; full probe sweep not the paper's mechanistic claims). Santhali
  (Ol Chiki) and Kashmiri (Perso-Arabic) are **zero-shot floor only** —
  no training pipeline — because those two languages are the published
  outliers the project exists to speak to.
- **Line manifests.** `export_line_manifest.py` +
  `export_manifest_scaled.py` (Decisions #37, #41, #43, #44). Canonical
  crop height **70 px** = 5 × ViT patch 14. Hindi natural: **2538**
  lines. Design target was 100 pages/mode.

### Scoring (Stage 0)

Order of operations is load-bearing: whitespace (Tier 0) → NFC (already
in olmOCR-style eval, Decision #4) → Tier 1 table → Tier 2
transliteration → human residual. Taxonomy **recomputes** Tier 1/2 live
from current code; notes only label residuals (Decision #35). Baselines
are append+skip, per-image OS-isolated timeouts, `--data-root` /
`--export-zip` (Decisions #31–#34, #42). PaddleOCR jsonl is complete
for the Stage 0 image set (420 rows; `docs/tier0e_paddleocr.md`).

### Renderer (Stage 1)

HarfBuzz shaping (uharfbuzz metrics + Pillow/raqm paint, Decision #27).
Do not hand-place conjuncts. Frequency modes: sentence-level importance
sampling was not enough; **bigram-guided synthesis** hits TV ≤ 0.08
(#29). Layout bank and degradation *source quality* are **PARTIAL**
(extractors exist; `form` / `table-embedded` and india.gov fetch
deferred; degradation pool is IA + GlotOCR degraded, not
prescriptions/forms). That is why Stage 3 (tau vs layout complexity) is
blocked.

### Instrument (Stage 2a)

Two models were a founding choice (Decision #1): a pretrained VLM
cannot isolate exposure. Vocabulary is grapheme clusters, not BPE (#2),
so "times this visual unit was seen" is not tangled with merge
frequency. Encoder 320-d / decoder 384-d with a linear bridge (#36).
Greedy decode, no beam, no KV cache, returns probe tensors (#38).
fp16 + grad checkpoint on Turing T4 (no bf16). Constant LR `3e-4`,
batch 32, 5000 steps, **last checkpoint**, three seeds. Realized size
**19,607,104** params at |V|≈367 (Decision #65; `docs/training_config.md`).
Checkpoints live on Colab Drive; jsonl in git is the evidence (#61).

Flattened and inverted runs exist and are used as negative controls
(they did not yield a usable reader). Paper mechanistic claims: natural
seeds 0–2 only (#66).

### Probes and analysis (Stage 4)

Each probe writes JSONL incrementally and skips completed ids. Analysis
scripts in `src/analysis/` write `docs/*_analysis.md`. Offline battery:
`paper_defensibility_stats.py` (Decision #64). Figures:
`make_paper_figures.py` → `paper/figures/` PDF and `docs/figures/` PNG.

**Not in git:** `data/raw/`, `data/predictions/`, `data/cache/`,
`checkpoints/`, `_local_archives/`. Probe jsonl **is** in git and must
be pushed immediately after a Colab run (AGENTS.md).

### Constraints that shaped the science

- Free Colab T4 only.
- Sarvam budget ~200 pages ever, ₹0.5/page, max 10 pages/job; cache
  first (Decision #19). Stage 5a used **35 pages / ₹17.50**.
- No fabricated Sarvam product facts; re-fetch docs.sarvam.ai / blog
  (AGENTS.md).
- No invented citations (Decision #67).
- Evaluation images are GlotOCR renders — label them that way (#74).

---

## Implementation status

Copied from `IMPLEMENTATION.md` in prose so this file stands alone.
When a checkbox flips, update **both** files.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done ·
`[!]` blocked. `[x] BUILT — QUEUED` = code ran on fake/smoke data.
`[x] BUILT — VERIFIED` = ran on real project data.

**Preprint scope:** Stages 0–2 and Stage 4 probes. **Stage 5b and Stage
6 are computed and documented offline** (post-preprint extensions; not
deferred — see reports below).

### Stage 0 — Error taxonomy

| Module | Status | Notes |
|---|---|---|
| `run_baselines.py` | `[x]` VERIFIED | Tesseract+Surya jsonl OK; PaddleOCR 420/420 scored 2026-09-12 (`docs/tier0e_paddleocr.md`) |
| `error_taxonomy.py` | VERIFIED | `error_taxonomy.csv`; UNREVIEWED large |
| `equivalence_tables.py` | VERIFIED | `__main__` 9/9 |
| `transliteration_equivalence.py` | VERIFIED | 38/38 validation pairs; corpus TIER2 0% after T1 |
| LLM-as-judge on Tier 2 disagreements | `[ ]` | Blocked until Tier 2 fires on real diffs |
| `hand_review.py` + assist | `[x]` | 13/13 + 7/7 self-tests |
| `adjudication_sample.py` | VERIFIED | n=200 seed=42; CI waits on `--queue` labels |

**Acceptance (open):** per-engine Tier 1 / Tier 2 / genuine fractions
as *final* after adjudication.

### Stage 1 — Renderer

| Module | Status |
|---|---|
| `layout_sources.py` | PARTIAL — no form/table-embedded; india.gov fetch flaky |
| `degradation_profile.py` | PARTIAL — measured, wrong source mix |
| `glyph_frequency.py` | `[x]` TV ≤ 0.08 |
| `render.py` | `[x]` HarfBuzz + GT boxes |
| Tiers A/B/C | `[x]` B inherits PARTIAL degradation sources |

### Stage 2a — Instrument

Tokenizer, encoder, decoder, `train.py`, `generate.py`: **VERIFIED** on
real Hindi (9 checkpoints). `make_fake_probe1_data.py`: QUEUED (that is
the fake path).

### Stage 2b — Demo

Loaders, LoRA config, pairwise orderer, SFT script, RLVR **reward**
wired. Decision #3 **closed** (#84): Colab T4 dummy LoRA selected
`ds4sd/SmolDocling-256M-preview` (1.63 GB). SFT 100 steps on that
adapter. RLVR **policy** training not run (`docs/tier2_rlvr_ablation.md`).
What a retrain would actually require: `docs/rlvr_scoping.md`.
Reports: `docs/tier2_stage2b_demo.md`.

### Stage 3 — Structure metrics

Kendall tau + table-binding **implemented and tested**. Geometric
baseline on the live bank (25/2/1/0/0 templates); empty buckets stay
holes (#81). No demo-model curve. `docs/tier2_stage3_reading_order.md`.

### Stage 4 — Probes

| Probe | Status | Artifacts |
|---|---|---|
| 1 exposure orchestrator | QUEUED / FE ran | β withheld; `docs/probe1_fixed_effects.md` |
| 2 confusion | VERIFIED | `probe2_hindi_natural_seed{0,1,2}.jsonl` |
| 3 blank/noise | VERIFIED | 9 files n=100 |
| 3b training curve | VERIFIED | 5 snapshots × 3 seeds |
| 4 equivalence on instrument | `[ ]` method exists | |
| 5 calibration | VERIFIED | 9 files n=100 |
| 5b zero-shot | VERIFIED | 720 records |
| Attention ablation | VERIFIED | 3 seeds; `docs/attention_ablation_analysis.md` |
| GT-likelihood | VERIFIED | 360 records |
| Paper stats + figures | VERIFIED | `docs/paper_defensibility_stats.md`; `paper/figures/` |
| mismatch TF, cross-attn norms | code; **not run** | `docs/tier0b_gt_mismatch.md`, `docs/tier0c_cross_attn_norms.md` |
| noise/scrambled GT-likelihood | VERIFIED 3 seeds | `docs/tier0d_noise_scrambled.md`; now in `paper/main.tex` |
| Probe 5 train-overlap AUROC split | VERIFIED (offline) | `docs/memorisation_split.md` — non-match n=0 |
| Step 4b KL / argmax vs 5-gram | VERIFIED 3 seeds | `docs/ngram_kl_argmax.md` |
| PaddleOCR same-protocol control | **not viable** | `docs/paddleocr_feasibility.md` |
| 6 synthetic–real (paper scope) | VERIFIED | 0 leakage |

### Stage 5 — Sarvam

| Piece | Status |
|---|---|
| `sarvam_client.py` | `[x]` Extract, cache by SHA-256 |
| `sarvam_transfer_probe.py` | VERIFIED RUN — 35 pages |
| `analyze_sarvam_transfer.py` | `[ ]` 5a bootstrap CIs |
| Stage 5b rank correlation | **computed / documented (offline)** — `docs/stage5b_rank_correlation.md`; not in preprint |

### Stage 6 — Cascade

**computed / documented (offline)** — `docs/stage6_triage_cascade.md`;
`cascade.py` ran threshold sweep on Stage 5 cache; not in preprint.

### Tooling

`make smoke-test` VERIFIED (architecture only, no findings).

---

## Decision catalog (summaries)

Full text stays in `DECISIONS.md`. This catalog is so you can search
this file for "why." "Why rejected" is compressed; if you are about to
override a choice, read the numbered entry first.

### Founding scientific choices (#1–#16)

**#1 Two models.** Instrument (scratch, zero Indic pretraining) vs demo
(LoRA on a small VLM). One LoRA model cannot isolate exposure:
pretraining already drowned the manipulation.

**#2 Grapheme vocab, not BPE.** Probe 1 measures exposure per visual
unit. BPE merges mix tokenizer frequency with visual frequency.

**#3 Demo base = SmolDocling-256M-preview.** Closed on a Colab T4
four-way dummy LoRA (#84): 1.63 GB peak, below granite (1.84 GB) and
both LightOnOCR ids (5.67 GB). #79 was the "do not fake T4" hold.

**#4 Do not re-solve NFC.** olmOCR-bench already NFC-normalizes. The
claim is about equivalences NFC *ignores*.

**#5** Reserved.

**#6 Script scope.** Hindi deep, Bengali structural, Santhali/Kashmiri
zero-shot only. Full four-script training was rejected on compute/fonts;
dropping the two outliers was rejected because they *are* Sarvam's
published spread (Kashmiri 55.93 / Santhali 80.32, re-verified on the
Vision blog).

**#7 Align at grapheme clusters.** Code-point CER double-counts matras
inside one akshara.

**#8 Tier 2 via ISO 15919.** Not a hand table (open-ended), not lossy
romanization, not LLM-as-primary-scorer (irreproducible).

**#9 Real layouts, not invented templates.** Internet Archive /
india.gov / Wikipedia. Bank still PARTIAL.

**#10 Naturalness confound.** Flattening frequency can destroy language
structure; bigram packing keeps local plausibility under a hard quota.

**#11 RLVR ablation = coverage term only.** One checkable claim: without
coverage, the policy omits text to game accuracy.

**#12 Tables = header-cell binding**, not full table-to-prose.

**#13 Compare structure to Sarvam Extract** (per-field confidence), not
Digitise.

**#14 Three seeds per Probe 1 condition.** Non-negotiable. Seed-0-only
headlines are forbidden. #53 later showed why: Kashmiri "significance"
died at seeds 1–2.

**#15 Transfer on Tier A and B.** Clean-only comparison is uninformative.

**#16 Cascade reports router quality**, not ₹ saved. Instrument ≉ better
than Tesseract.

### Taxonomy and baselines (#18–#35, #42, #54–#55)

**#18** Tier 2 = strict phonetic identity, not conventional spelling
variance. **#19** Cache every Sarvam page once. **#20** Assist module
separate from the viewer so notes record confirm vs override. **#21**
PDF text-layer for born-digital layouts; ink projection for scans.
**#22** Tier 0 whitespace before assist diffs. **#23** Assist false
positives were punctuation gaps → grow Tier 1, don't relabel as
order/matra. **#26** Pipe-as-danda and space-before-punctuation in
Tier 1. **#28 / #42** Surya `recognition` API; PaddleOCR 3.x
`predict()`, no `show_log`, MKLDNN off on CPU. **#31–#34** Append+skip
resume, `--data-root` export, per-image hard timeouts with OS-process
kill. **#35** Recompute T1/T2 live. **#54** Honest 38-pair validation;
do not invent pairs to pad n. **#55** Stratified UNREVIEWED sample +
bootstrap CI after humans label.

### Renderer internals (#21, #24–#30, #41, #44)

**#24** Degradation parameters inverted from measurements, not
hardcoded. **#25** Sentence-level IS over Indic clusters. **#27**
uharfbuzz + Pillow/raqm; Tier A is clean. **#29** Bigram-guided
synthesis to hit TV ≤ 0.08. **#30** Audit: don't mark layout/degradation
complete when they aren't. **#41** Line boxes from `measure_line_width`.
**#44** Height 70 px.

### Instrument and Colab (#32, #36–#40, #43, #47–#48, #65–#66, #73)

**#32** One data root; export unpacks into IMPLEMENTATION paths. **#36**
320/384 bridge in `train.py`. **#37** Line-crop manifests, not full
pages, for Probe 1 speed. **#38** Greedy, no KV cache. **#39** Probe 1
in-process; checkpoint at `total_steps` = done. **#40** `make smoke-test`
is architecture proof only. **#43** Scaled export uses `--data-root`.
**#47** Script-scoped checkpoint/tokenizer paths. **#48**
`--keep-snapshots` for Probe 3b (default resume overwrites). **#65**
Paper Section 3 reports the *trained* instrument, not a 30–60M design
target. **#66** Mechanistic claims = three natural seeds. **#73** No
eval-string holdout during training; n-gram analysis excludes exact
string overlaps after the fact.

### Probe statistics and paper (#45–#46, #49–#53, #56–#64, #67–#89)

**#45 / #58** Probe 6: resize Tier C to 70 px; paper scope is Tier C +
blank, not the full synthetic gap. **#46** Methodology upgrades listed
after the Conclusion — several later landed as offline stats (#64).
**#49** Withhold Probe 1 β at accuracy floor. **#50** No CER on unseen
scripts (tokenizer cannot emit those code points). **#51–#53** Bonferroni
on 5b; TOST δ=0.05; retract seed-0 Kashmiri pass. **#56** Ablate by
zeroing encoder memory; KL on shared *full-memory* prefixes (not GT
prefixes — see `docs/training_config.md`). **#57** Probe 2 is
GT-aligned + *p*(true). **#59** Extract, not Digitise, for confidence.
**#60** Cite Sarvam blog with fetch date. **#61** Canonical results =
`data/probe_results` + `docs/`; Colab zips stay local. **#62**
Teacher-forced log *p*(GT) vs max-softmax bias. **#63** Follow-up
probes (mismatch, cross-attn norms authored; noise/scrambled **ran**;
mismatch/cross-attn still blocked on local checkpoints). **#64** Unified figure generator +
offline defensibility battery. **#67** Live bibliography, no invented
venues. **#68** Single `paper/` directory. **#69** GitHub Pages: one
URL, two modes. **#70–#71** Figure 1 inset layout. **#72** Surya
positive-control citation is a reported preprint, not our verification.
**#74** Never call eval images "real." **#75** Preprint tab matches tex
terminology. **#76** README is a map. **#77** One Sarvam-facing note.
**#78** Paddle in-process fill. **#79** Do not close demo-base without
T4 VRAM. **#80** Demo SFT = Hindi natural line crops. **#81** Stage 3
scores live bank region order; no invented tables. **#82** RLVR reward
yes, train no until SFT. **#83** Mistral3 dummy image-token ids.
**#84** Close Decision #3 on T4: SmolDocling-256M.
**#85** Step 4b stores per-step KL + argmax flags, not full softmax jsonl.
**#86** PaddleOCR rec is CTC/SVTR, not an AR grapheme decoder — no instrument-matched control.
**#87** Project page: preprint first; two short modes.
**#88** RLVR accuracy term is emitted-only, not GT-length char_acc.
**#89** Stage 5b: Spearman ρ, 10k permutation null, per-image unit;
locked before looking at ρ. Spend gated.

---

## First principles: the central finding

Read this section before the teaching chapters if you want the
mechanistic story without opening code. Every number below cites a
committed doc or the preprint — do not invent headlines.

> **Framing (repeat):** The paper does **not** prove that a model which
> can read is hallucinating. It shows that, in this from-scratch
> instrument, **confidence remains extremely high even when the model
> demonstrably does not read the evaluation image.**

### What 0.90 confidence actually means

At each decoding step the instrument outputs a probability vector over
its vocabulary — roughly **367 grapheme clusters** (Decision #65). After
softmax, the confidence scalar we report is:

\[
C_t = \max_v P(v \mid \text{image}, y_{<t})
\]

That is the peak of the distribution at step *t*, not the probability
assigned to the ground-truth token.

Important subtlety: when we say the model is "90% confident about token
A," **token A is not one fixed grapheme**. It is whatever wins the argmax
at that step — and that identity **varies across images**. The shock is
not that one wrong grapheme dominates globally; it is that **whatever
wins is almost never the correct first grapheme**. Pooled over three
seeds (60 images × 3 = 180 sequences), ground truth is the argmax at
position 0 in **0 / 180** sequences
([`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md),
Abstract Fact).

Meanwhile self-generated max-softmax at the first generated step stays
≈ **0.9020** on text-bearing inputs and ≈ **0.8952** on **blank**
inputs, while teacher-forced *p*(GT) at position 0 is ≈ **2.204×10⁻¹¹**
(real) and ≈ **6.328×10⁻¹¹** (blank).

| Signal | Text-bearing | Blank |
|---|---:|---:|
| Self-generated max-softmax (pos 1) | **0.9020** | **0.8952** |
| Teacher-forced *p*(GT) at pos 0 | **2.204×10⁻¹¹** | **6.328×10⁻¹¹** |
| GT is argmax at pos 0 | **0 / 180** | **0 / 180** |

That is the **dissociation**: extremely high confidence about *some*
token, effectively zero probability on the visually correct one — and
blank behaves similarly.

### Sharp vs uncertain — not the same failure mode

These are different shapes of softmax output:

- **Uncertain:** probabilities are relatively flat — no single token
  dominates (uniform over |V| ≈ 367 would give ≈ **0.0027** each).
- **Confidently wrong:** one token gets ≈ **0.90** while *p*(GT) ≈
  **10⁻¹¹** — a very sharp peak on the wrong answer.

The instrument is in the second regime. Position-0 predictive entropy
is **0.333 nats** pooled (Table E1, [`paper/main.tex`](./paper/main.tex))
— low enough to confirm the distribution is peaked, not flat. A
scale-free companion statistic: median GT rank **70.5 / 367** (IQR
[33.0, 136.2]), with **35.0%** of sequences ranking the correct symbol
below position 100 in the model's own distribution (same table). Do **not**
anchor the headline on uniform 1/|V| alone (SITE_SYNC_PROMPT): the point
is sharp wrong confidence, not "below chance."

### Why position 0 isolates vision — but we measured the full profile

**Autoregressive** decoding emits one grapheme at a time; each step can
depend on the image and on everything already generated (`<BOS> → भ →
ा → …`). At step *k* > 0, a strong **language prior** can mask a weak
visual signal: "after `भा`, the next cluster is probably `र`."

At **position 0** there is no Hindi prefix — only `<BOS>`. Only the
image can supply input-dependent information for the first grapheme.
That makes position 0 the cleanest grounding diagnostic.

We did not stop at position 0. Teacher-forced mean log *p*(GT) by
position bucket, instrument vs text-only 5-gram
([`docs/position_matched_ngrams.md`](./docs/position_matched_ngrams.md)):

| Pos | Instrument (real) | 5-gram |
|---|---:|---:|
| 0 | **−24.54** | **−4.73** |
| 1 | **−8.12** | **−1.14** |
| 2–9 | **−0.48** | **−0.31** |
| 10–19 | **−0.15** | **−0.26** |
| 20–39 | **−0.27** | **−0.20** |
| 40+ | **−6.12** | **−0.18** |

Narrative read: position **0** is catastrophic (~**19.8 nats** below
the text-only first-grapheme reference); position **1** is still bad;
positions **2–39** look language-prior-like (within a few tenths of a
nat of a 5-gram on the GT token); position **40+** diverges again (long
tail / EOS effects). Full-softmax KL(model ‖ 5-gram) and argmax agreement
confirm mid-sequence overlap ([`docs/ngram_kl_argmax.md`](./docs/ngram_kl_argmax.md),
pooled): KL **0.27–0.32** nats on positions 2–39, argmax agreement
**91–94%**.

### Language prior — not "world knowledge"

Safer wording: the decoder learned **statistical regularities of the
training text** — local Indic grapheme patterns like `भा → र` — the same
mechanism a text-only n-gram captures, not encyclopedic facts. Probe 5
evaluates on strings drawn from the **training manifest** (same text
regime the decoder saw); held-out-image AUROC ≈ **0.5698** shows
in-manifest AUROC **0.838** does not generalize
([`docs/memorisation_split.md`](./docs/memorisation_split.md),
[`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)).

The central question: **is confidence coming from visual evidence or
from the decoder's ability to produce statistically plausible text?**

### Blank images — careful phrasing

Blank controls do **not** prove the model "learned vision and then
ignored it." An equally live alternative: it **may never have learned a
useful visual representation** that transfers to evaluation renders.
The paper keeps both explanations alive. What blank *does* show: high
confidence and near-zero *p*(GT) persist when meaningful visual content
is removed — same dissociation as text-bearing inputs.

### Encoder ablation — shape moves; peak barely does

Zeroing encoder memory before `memory_projection` changes the output
distribution (mean KL(full ‖ zero) ≈ **1.075** nats; top-1 agreement ≈
**0.879**; prior sufficiency ≈ **0.8827** — ~**8%** of steps flip
argmax and carry ~**97%** of KL), but pooled mean max-softmax is almost
unchanged (full **0.9861** vs zero **0.9891**, Δ **−0.0030**). The
visual pathway **does** affect distribution shape and some token choices;
the max-softmax scalar is largely insensitive. Do **not** say "the model
doesn't use vision" — say confidence peak is not tracking visual
contribution. Source:
[`docs/attention_ablation_analysis.md`](./docs/attention_ablation_analysis.md),
[`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md) Item 5.

### Vocabulary — why grapheme `\X`, not BPE

The scientific question is visual exposure at **grapheme** granularity.
BPE subword merges would entangle "how often the model saw a visual unit"
with tokenizer statistics. The instrument segments with Unicode grapheme
cluster rules (`\X`), keeps clusters with frequency ≥ 5 in the training
manifest, and ends at |V| ≈ **367** (Decision #65;
[`docs/training_config.md`](./docs/training_config.md)). Tier **0**
whitespace normalization, Tier **1** encoding-equivalence tables, and
Tier **2** assist diffs (Stage 0) sit upstream of any error rate.

### Probe 1 — dial worked; causal β withheld

The exposure **manipulation** worked: realized grapheme distributions
within TV ≤ **0.08** (natural **0**, flat ≈ **0.047**, inv ≈ **0.005**
— [`docs/RESULTS.md`](./docs/RESULTS.md)). Sentence-level selection alone
could not hit flattened/inverted targets; bigram-guided synthesis was
used instead (Decision #28). Line accuracy: natural **18.3%**, flattened
**0.33%**, inverted **0.67%**
([`docs/probe1_fixed_effects.md`](./docs/probe1_fixed_effects.md)) —
floor performance, so no exposure coefficient β was estimated (Decision
#49).

### Three seeds — because seed 0 failed to replicate

Probes and training report three seeds (0, 1, 2) after a single-seed
Kashmiri headline did not replicate (Decision #56). Do not regress to
seed-0-only numbers in new write-ups.

### Calibration — ECE and the AUROC trap

Natural Hindi (synthetic manifest): mean confidence ≈ **0.994**, line
accuracy ≈ **0.183**, ECE ≈ **0.811**
([`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)).
Pooled AUROC **0.838** ranks correctness **within the training-manifest
regime only**. Held-out GlotOCR-render AUROC(conf → CER < median) ≈
**0.5698** — near chance.

### Where this section hands off

- **Chapters 0–3** — CV basics, scoring, renderer, instrument architecture.
- **Chapter 7** — full probe suite with paths and interpretation.
- **Chapters 8–9** — Sarvam transfer and triage cascade.
- **Presenting to Sarvam** — pitch order for product conversations.
- **Experiments done · Not proved · Roadmap** — checklist after the Sarvam pitch.
- **Appendix H** — verbatim interview Q&A.

---

## Related work

This project did not discover in isolation that multimodal models can
lean on language priors. It provides a **controlled, grapheme-level Indic
OCR diagnostic** — blank/unseen-script/ablation/position-0 instruments
that are hard to run on a closed API. Verified papers to cite:

| Work | What it contributes | How we differ |
|---|---|---|
| [M3ID — Favero et al., CVPR 2024](https://arxiv.org/abs/2403.14003) | **Conditioning dilution**: autoregressive VLMs rely more on language prior as generation proceeds; M3ID boosts visual grounding during decode. | They **mitigate** grounding decay in large VLMs; we **diagnose** position-resolved confidence vs *p*(GT) in a small from-scratch OCR instrument. |
| [First Logit Boosting — Ha et al., CVPR 2026](https://arxiv.org/abs/2604.00455) | First token as **visual anchor** in models that read; reuse first-logit signal to reduce later hallucination. | They find strong position-0 grounding in capable models; we find catastrophic position-0 failure in our instrument — motivates a **positive control** under the same protocol. |
| [BICR — Grounded or Guessing?](https://arxiv.org/abs/2605.10893) | Confidence not grounded in image; **real vs blacked-out** contrast for LVLM confidence estimation. | BICR learns a probe on hidden states in large VLMs; we provide a **training-free, position-specific** OCR diagnostic with teacher-forced *p*(GT). |
| [HALP — EACL 2026](https://aclanthology.org/2026.eacl-long.287/) | Pre-generation **hallucination risk** from internal representations (reported high AUROC on some VLMs). | HALP targets general VLM hallucination detection; we motivate **grounding-aware OCR confidence** (real-vs-blank likelihood gap, cross-attention contribution). |
| [GlotOCR Bench](https://arxiv.org/abs/2604.12978) | Multilingual OCR benchmark; models struggle beyond a handful of scripts; unfamiliar scripts → wrong-script emissions. | We use GlotOCR text and eval **renders**; our contribution is the **instrument + probes**, not a new benchmark score. |

Say in interviews: existing work established ungrounded confidence and
multilingual OCR weakness; **this repo asks what that failure looks like
when you can manipulate exposure and intervene on the visual pathway.**

---

## Presenting to Sarvam

Use this order in a product or research call. Stage 5 is **not** "our
20M model vs Sarvam accuracy."

### Motivation (production phenomenon)

Sarvam Vision reports substantial language-wise spread on its Indic OCR
benchmark — Hindi **95.91%**, Santhali **80.32%**, Kashmiri **55.93%**
word accuracy ([sarvam.ai/blogs/sarvam-vision](https://www.sarvam.ai/blogs/sarvam-vision);
Decisions #6, #60). Production pipelines still expose confidence for
routing. The question: **does that confidence reflect difficulty regimes
the benchmark already shows?**

### What we built (instrument → mechanism)

We could not run counterfactuals on an API (blank encoder, full softmax,
position-0 *p*(GT), controlled training exposure). So we built:

```text
Stage 0 — Indic-aware scoring
Stage 1 — controlled renderer (exposure dial)
Stage 2a — ~19.6M from-scratch instrument
Stage 4 — probes (blank, unseen script, ablation, teacher forcing, n-gram)
```

Main mechanistic result: max-softmax ≈ **0.90** while *p*(GT) at
position 0 ≈ **2.2×10⁻¹¹**, GT never argmax — **in this instrument
that does not read held-out images** (CER ≈ **0.985** text vs ≈ **0.949**
blank — not meaningfully different; [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)).

### External validation (not proof about Sarvam's encoder)

**Stage 5a:** Extract page confidence Hindi **0.9997** → Kashmiri
**0.9970** (Δ **0.0027**) across a **39.98 pp** published accuracy gap;
blanks **0.0000** ([`docs/sarvam_vision_confidence.md`](./docs/sarvam_vision_confidence.md)).

**Stage 5b:** Per-image instrument difficulty vs Sarvam CER on Hindi
plains (n=60): Spearman ρ = **0.0293**, permutation p = **0.8267** —
pre-registered null ([`docs/stage5b_rank_correlation.md`](./docs/stage5b_rank_correlation.md)).

**Stage 6:** At 20% escalation (k=12), instrument-confidence routing
residual CER **0.0109** vs random **0.0095**, Tesseract **0.0079**
([`docs/stage6_triage_cascade.md`](./docs/stage6_triage_cascade.md)).

These show **no useful transfer** of our instrument's confidence/difficulty
signal to Sarvam in this sample — not that Sarvam "doesn't read."

### Contribution pitch (what to say)

> "We built a controlled OCR instrument to decompose confidence into
> visual evidence versus language-prior behavior — blank controls, unseen
> scripts, encoder ablation, position-resolved teacher-forced likelihood,
> and comparison to text-only n-gram models. Max-softmax could remain near
> ceiling while the correct visual token had effectively zero probability.
> We then tested transfer to Sarvam Vision at language and per-image
> levels and found no useful transfer in our sample. The production
> takeaway is not 'confidence is useless' but that **max-softmax is not
> necessarily a grounding signal** — the next step is grounding-aware
> confidence (real-vs-blank representation contrast, image-conditioned
> *p*(GT), cross-attention contribution — cf. BICR, HALP)."

### What not to say to Sarvam

- "We proved VLMs don't look at images."
- "Our model vs yours on accuracy."
- "Sarvam confidence is broken" (we tested a small Extract sample; blanks
  did go to zero).

### What we'd do next (with their context)

1. Held-out synthetic train/val/test split before training.
2. Positive control — known-good Devanagari reader, same position-0 protocol.
3. Grounding score: *p*(GT | image) − *p*(GT | blank), not max-softmax alone.
4. Larger matched Sarvam evaluation (500–2000 pages), stratified by script.

---

## Experiments done · Not proved · Roadmap

Orientation lists only — cite docs for numbers; flip `IMPLEMENTATION.md`
when a roadmap item is built and verified.

### (A) Experiments already done

1. **Stage 0 — Indic-aware evaluation.** Tier 0 whitespace, Tier 1
   encoding equivalence, Tier 2 assist diffs, error taxonomy before
   comparing engines (`docs/error_taxonomy.csv`, Chapter 1).
2. **Stage 1 — Controlled renderer.** Natural / flattened / inverted
   exposure modes; TV dial verified (natural **0**, flat ≈ **0.047**,
   inv ≈ **0.005** — [`docs/RESULTS.md`](./docs/RESULTS.md)).
3. **Stage 2a — From-scratch instrument.** ~19.6M params, empty pretraining
   history, full softmax + teacher forcing
   ([`docs/training_config.md`](./docs/training_config.md)).
4. **Probe 1 — Exposure manipulation.** Dial worked; causal β withheld
   at accuracy floor ([`docs/probe1_fixed_effects.md`](./docs/probe1_fixed_effects.md)).
5. **Probe 2 — Confusion / alternative tokens.** What the model considers
   besides GT ([`docs/probe2_confusion_analysis.md`](./docs/probe2_confusion_analysis.md)).
6. **Probe 3 — Blank and matched noise.** Confidence invariance when
   visual content removed ([`docs/probe3_curve_analysis.md`](./docs/probe3_curve_analysis.md)).
7. **Probe 3b — Training-curve snapshots.** Accuracy vs step along natural
   training (same doc — [`docs/probe3_curve_analysis.md`](./docs/probe3_curve_analysis.md)).
8. **Probe 5 / 5b — Calibration and unseen scripts.** ECE ≈ **0.811**;
   Ol Chiki / Perso-Arabic zero target-script emissions
   ([`docs/results_analysis.md`](./docs/results_analysis.md),
   [`docs/probe5b_analysis.md`](./docs/probe5b_analysis.md)).
9. **Encoder ablation.** Zero visual memory: Δ confidence **−0.0030**,
   KL **1.075** nats ([`docs/attention_ablation_analysis.md`](./docs/attention_ablation_analysis.md)).
10. **GT likelihood by position.** Teacher-forced log *p*(GT) + entropy;
    position-0 dissociation ([`docs/gt_likelihood_analysis.md`](./docs/gt_likelihood_analysis.md),
    [`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)).
11. **Position-matched n-grams + KL/argmax (Steps 4a/4b).** Text-only
    baseline vs full decoder softmax
    ([`docs/position_matched_ngrams.md`](./docs/position_matched_ngrams.md),
    [`docs/ngram_kl_argmax.md`](./docs/ngram_kl_argmax.md)).
12. **Tier 0d — Noise and patch-scramble.** Same likelihood pattern as
    blank/text ([`docs/tier0d_noise_scrambled.md`](./docs/tier0d_noise_scrambled.md)).
13. **Stage 5a — Sarvam Extract vs published spread.** Δ confidence
    **0.0027** across **39.98 pp** accuracy gap
    ([`docs/sarvam_vision_confidence.md`](./docs/sarvam_vision_confidence.md)).
14. **Stage 5b + 6 — Transfer and triage.** ρ = **0.0293**, p = **0.8267**;
    instrument routing worse than random at 20% escalation
    ([`docs/stage5b_rank_correlation.md`](./docs/stage5b_rank_correlation.md),
    [`docs/stage6_triage_cascade.md`](./docs/stage6_triage_cascade.md)).

### (B) What we have NOT proved

1. That **models which can read** universally hallucinate or ignore images.
2. That **this instrument "doesn't use vision"** — ablation moves
   distributions; only the confidence peak is insensitive.
3. That **Sarvam doesn't look at images** — we tested a small Extract
   sample; we cannot ablate their encoder.
4. That **grapheme exposure causally improves OCR** — Probe 1 β withheld
   (flattened **0.33%** / inverted **0.67%** vs natural **18.3%**).
5. That the model **learned vision then ignored it** vs **never learned
   useful visual features** — both remain live (blank controls).
6. That **renderer domain shift alone** explains held-out CER ≈ 1 — noise,
   scramble, and ablation favor a bypass account but do not prove it
   ([`docs/probe6_synthetic_real_analysis.md`](./docs/probe6_synthetic_real_analysis.md)).
7. That **AUROC 0.838 generalizes** — it is in-training-manifest only;
   held-out-image AUROC ≈ **0.5698**.
8. That **instrument difficulty predicts Sarvam per-image CER** — primary
   null at n=60 (Stage 5b).
9. That **max-softmax suffices for production routing** — Stage 6 shows
   worse-than-random escalation at the fair operating point.

### (C) Next-round roadmap (priority order)

1. **Positive control first.** Known-good autoregressive Devanagari reader
   under the same position-0 / blank / ablation / GT-likelihood protocol
   ([`docs/paddleocr_feasibility.md`](./docs/paddleocr_feasibility.md),
   [`docs/surya_positive_control.md`](./docs/surya_positive_control.md)).
2. **Per-position curve as primary exhibit.** Full teacher-forced profile
   (already computed — make it the default slide after position 0).
3. **Vocabulary audit.** Confirm `\X` clusters, rare-grapheme handling,
   and tokenizer–eval alignment did not confound exposure counts.
4. **Natural exposure redesign.** Reweight **real sentences** instead of
   flattened/inverted synthetic language (mixed-effects target).
5. **Held-out synthetic split before training.** Train / val / test text
   disjoint; never touch test until final eval — fixes AUROC/calibration
   interpretability.
6. **Confound checklist A–F.** When comparing conditions, log glyph
   complexity, exposure count, font, sentence context, layout, and image
   quality separately — do not let frequency manipulation change
   everything at once.
7. **Train longer — after design fixes.** More steps/data only once split,
   exposure, and eval protocol are fixed; not "same broken experiment, more
   GPU."
8. **Grounding-aware confidence.** *p*(GT | image) − *p*(GT | blank),
   cross-attention contribution, or hidden-state probes (BICR / HALP
   direction) — replace max-softmax as the routing signal.
9. **Larger Sarvam evaluation.** 500–2000 matched pages, stratified by
   script, to test whether any grounding score predicts production errors
   beyond n=60.

---

## Chapter 0 — What Is Computer Vision, and Why Is Reading Hard For a Machine

> **Start here for CV basics.** If you already want the central
> mechanistic finding (position 0, n-grams, ablation), read **First
> principles: the central finding** above first — then return here for
> Indic-script motivation.

### Start with what a computer actually sees

A photograph, to you, is a scene: a page of a book, a signature, a road
sign. To a computer, it is a grid of numbers. A single-color pixel is
three numbers — how much red, how much green, how much blue, usually
each from 0 to 255. A modest photo, say 1000 pixels wide and 1000 tall,
is three million numbers. There is no “page” in there, no “letter,” no
“word.” Just a grid of brightness values.

Computer vision is the field concerned with one question: how do you get
from that grid of numbers to something that counts as understanding —
“this says Priya,” “this is a cat,” “this door is open”? Every technique
in this repo, from a ~20-million-parameter instrument model to a
production document system, is ultimately an answer to that one
question, just at different scales of ambition.

### Why “just teach it letters” doesn’t work

The oldest and most tempting approach to reading text from an image is:
find each letter, look up what it is, done. This is called
segmentation-then-classification, and for a long time it is roughly how
OCR worked. It fails for reasons that are genuinely instructive:

- **Letters touch each other.** In cursive handwriting, in many fonts,
  and structurally in scripts like Devanagari (where a connecting line
  called the *shirorekha* runs across the top of a whole word), there
  often isn’t a clean gap between one letter and the next to cut along.
- **One “letter” isn’t always one visual unit.** In Devanagari, a base
  consonant can combine with a vowel sign (a *matra*) that appears
  above, below, beside, or wrapped around it, and two or more
  consonants can stack into a single fused glyph called a *conjunct*
  (a *saṃyuktākṣara*). The visually atomic unit — the thing a reader’s
  eye treats as one character — is called a *grapheme cluster*, and it
  can correspond to two, three, or more separate values in the
  underlying digital text encoding. This mismatch between “one visual
  thing” and “one encoded thing” turns out to matter enormously for
  how you measure whether a system got the reading right — see
  Chapter 1.
- **Context changes what a shape means.** The same rough pixel pattern
  can be one letter in one font and a different letter in another; a
  smudge can look exactly like a diacritic. A system that classifies
  each cropped-out shape independently, with no knowledge of what came
  before or after it, throws away the single strongest signal a human
  reader actually uses: everything else on the page.

Modern systems, including the small one built in this repo, sidestep
segmentation almost entirely. Instead of “find each letter, then
classify it,” the approach is “look at the whole image, and generate the
text one unit at a time, using both the image and everything generated
so far as context.” That is a genuinely different kind of machine, and
it is worth understanding why it works.

### The two halves of a document-reading model

Every model in this repo, and production systems like it, has (at least)
two halves that do fundamentally different jobs.

**The encoder** looks at the image and turns it into a set of numeric
*features* — vectors that capture “what’s visually going on here,”
without yet committing to any specific letter or word. Modern encoders
for this job are usually Vision Transformers (ViTs): the image gets cut
into small square patches (say 14×14 pixels each), each patch becomes a
vector, and a mechanism called *self-attention* lets every patch’s
representation get updated based on every other patch. This is what
lets the model notice, for instance, that a mark above a letter and the
base letter below it belong together as one grapheme cluster, even
though they are in different patches.

**The decoder** takes those image features and generates text, one
grapheme cluster at a time. At each step it looks at the image features
and everything it has generated so far, and predicts what comes next.
This is the same basic mechanism that powers large language models, just
conditioned on an image instead of, or in addition to, prior text.

The connective tissue between these two halves — how image features get
translated into something the text decoder can use — is itself a design
decision with real consequences. In this project it is a simple linear
projection between two different hidden sizes (Chapter 3).

### Why this is a genuinely hard problem, not a solved one

English-language OCR on clean printed text is close to a solved problem
and has been for years. It is tempting to assume “OCR” in general is
solved and everything past that is refinement. This project exists
because that assumption is wrong for the majority of the world’s
scripts, and the reasons are specific rather than vague:

1. **Data.** The internet, and therefore the training data every large
   model learns from, is overwhelmingly English and a handful of other
   high-resource languages. A model can only get good at reading what
   it has seen enough of.
2. **Script structure.** Complex scripts genuinely require more visual
   reasoning per character than Latin text does — conjuncts, stacking
   diacritics, and connecting strokes are not just “different letters,”
   they are a different geometric problem.
3. **These two causes get tangled together in every published result.**
   When a benchmark shows a language scoring low, that low score could
   be because the model barely saw that language during training, or
   because that script is intrinsically harder to read, or some mix of
   both — and a benchmark table alone cannot tell you which. Published
   Indic OCR tables show large spreads across languages with no
   explanation of how much is which cause. (This project’s motivating
   numbers were re-checked against Sarvam’s own Vision blog in August
   2026; see `DECISIONS.md` #6.)

That untangling — how much of a language’s poor performance is “we
didn’t show the model enough of it” versus “this script is inherently
harder to read” — is not answerable by looking at outputs alone. It
requires being able to control what a model sees during training and
then measure what it learned. That requires *owning* the model, not
querying it through an API. That single sentence is the entire reason
this project builds a model from scratch rather than only ever calling
someone else’s.

### What’s coming

Every chapter after this one follows the same shape: a real problem in
computer vision, why it’s hard, and then what building a piece of this
repo taught about it. The scientific arc is: first make “wrong” mean
wrong (Chapter 1), then control what the model sees (Chapter 2), then
own a model with empty Indic history (Chapter 3), then interrogate
whether its confidence is visual grounding or language prior
(Chapter 7) — with production transfer and triage in Chapters 8–9.

> **What to remember.** A machine does not see letters; it sees numbers.
> Reading systems generate text from images one visual unit at a time,
> and for Indic scripts the hardest scientific question is often not
> “how do I get a higher score,” but “does that score — or that
> confidence — mean the image was actually read?”

---

## Chapter 1 — Measuring Correctness Is Its Own Hard Problem

Before asking whether confidence reflects vision, we need "wrong" to mean
**misreading**, not a different valid Unicode spelling. See **First
principles** §4–5 for why this precedes the position-0 story.

### The question that forced this chapter

Suppose two OCR engines disagree with the ground-truth string. Are they
both wrong? Or did one of them write the *same reading* using a
different, equally valid Unicode spelling?

If you cannot answer that, every published “error rate” for Indic OCR
is partly fiction — and every later claim about confidence or exposure
rests on sand. Stage 0 exists to force that answer into the open
*before* anyone trains a new model or trusts a confidence meter.

### What “correct” even means

Character error rate (CER) is the usual OCR metric: how many edits
(insertions, deletions, substitutions) does it take to turn the
prediction into the reference? That metric quietly assumes the
reference has one canonical digital form.

Indic orthography violates that assumption constantly. The same spoken
word, and often the same visual page, can be encoded several ways:

- Sentence-final punctuation as Devanagari danda `।`, Latin period `.`,
  or even a plain ASCII pipe `|` that some engines emit when they “see”
  a vertical stroke.
- The word *Hindi* as `हिन्दी` (explicit nasal consonant) or `हिंदी`
  (anusvara) — same pronunciation, classical sandhi, two spellings.
- Zero-width joiners and non-joiners around a virama, which change how
  a conjunct is *drawn* without changing what was *read*.
- Digits in Devanagari or Bengali script versus ASCII `0`–`9`.

Unicode already has a composition form called **NFC** that collapses
many “same character, different bytes” cases. Production eval harnesses
(including olmOCR-bench, which Sarvam’s own scoring wraps) already apply
NFC. So the interesting claim is **not** “nobody normalizes Unicode.”
It is: **NFC leaves a whole class of orthographic equivalences
untouched**, and those equivalences are real enough that counting them
as errors will make every engine look worse than it is.

This project therefore splits “not an error” into two tiers on purpose:

- **Tier 1 — encoding equivalence.** Deterministic, uncontroversial.
  Same reading, different bytes. Implemented as a hand-curated
  normalization table plus anusvara-sandhi rules in
  `src/eval/equivalence_tables.py`.
- **Tier 2 — phonetic equivalence.** A judgment call: are these two
  differently spelled strings “the same word”? Implemented by
  transliterating both into ISO 15919 Latin with aksharamukha and
  comparing there (`src/eval/transliteration_equivalence.py`). Reported
  *separately* from Tier 1, because Tier 2 can be wrong or arguable in
  ways Tier 1 cannot.

Whatever is left after those two passes is what a human should actually
look at. The residual labels this project uses are fixed:
genuine-misread, dropped-matra-nukta, reading-order-break, and
hallucinated-repeated-text. If none of those fit, the assist module is
allowed to say so rather than force a bucket.

### Why grapheme clusters, not code points

A single visual akshara can span several Unicode code points. If you
score at the code-point level, one wrong matra can look like multiple
errors, or a conjunct can be mis-attributed across pieces that were
never separate visual objects. So from Stage 0 onward, alignment and
counting happen at the **grapheme-cluster** level — the same unit the
instrument model will later treat as one vocabulary token. That choice
is Decision #7, and it is also why Chapter 0 spent time on “one visual
thing ≠ one encoded thing.”

### What we built, and why each piece exists

**1. Get real (image, text) pairs.**  
`src/data_pipeline/fetch_glotocr.py` pulls GlotOCR-bench lines for Hindi,
Bengali, Santhali, and Kashmiri into `data/raw/{language}/`, with both a
clean `*_plain.png` and a degraded `*_degraded.png` per id. Stage 0 needs
ground truth that did not come from the engines you are judging.

**2. Run existing engines without pretending they are the product.**  
`src/eval/run_baselines.py` runs Tesseract, Surya, and PaddleOCR and
writes one JSONL line per image under
`data/predictions/{engine}/{language}.jsonl`. It does not score. It only
produces the “predicted” half of the pair. It also has to survive Colab
and laptop interruptions: append+skip resume, per-image hard timeouts,
and a zip export that unpacks into the same paths the rest of the repo
expects. Those are engineering details, but they exist for a scientific
reason — a batch that silently rewrites or hangs is indistinguishable
from “we never measured this.”

**3. Explain away what is not an error.**  
Tier 0 collapses whitespace (line-wrapping noise is layout, not
reading). Tier 1 and Tier 2 then run in that order. The hand-review
viewer (`hand_review.py`) skips anything those tiers already explain,
so human attention goes to the unexplained cases. A separate assist
module (`hand_review_assist.py`) *suggests* a residual label; the human
still has to confirm, override, or skip. Auto-accepting suggestions
would turn the notes file into an agent artifact pretending to be a
hand taxonomy.

**4. Produce the Stage 0 report.**  
`src/eval/error_taxonomy.py` walks every prediction, recomputes Tier 1/2
live against current code (so fixing the equivalence table does not
require redoing the whole hand pass), and falls back to human labels
only for residuals. The deliverable is
`data/predictions/error_taxonomy.csv` plus a printed per-engine table:
what fraction of predictions are exact, Tier 1, Tier 2, genuine, or
still unreviewed.

```mermaid
flowchart LR
  GT[Ground truth] --> N[Whitespace + NFC]
  P[Engine prediction] --> N
  N --> T1[Tier 1 encoding]
  T1 -->|same| OK[Exact or Tier 1]
  T1 -->|different| T2[Tier 2 phonetic]
  T2 -->|same| PH[Tier 2]
  T2 -->|different| H[Human residual label]
```

### What the evidence shows

**Measured** from this checkout by running `python3 src/eval/error_taxonomy.py`:

On Tesseract’s 180 scored predictions, 28 were exact matches after
normalization (15.6%). Of the 152 that were *not* exact, **31 (20.4%)
were Tier 1 encoding variants** — same reading, different bytes — not
genuine misreads. Thirteen carried a human-confirmed genuine-misread
label; 108 were still UNREVIEWED because the hand pass has not covered
them yet. So the headline “about one in five apparent Tesseract errors
isn’t a real error” is real, and the report is also honestly
**incomplete** until UNREVIEWED shrinks.

Surya, on a larger set in this checkout (n=222), is exact much more
often (about 47%), and about 17% of its non-exact rows are still Tier 1.
PaddleOCR full corpus was filled 2026-09-12 (`docs/tier0e_paddleocr.md`):
n=420 scored, exact 11 (2.6%), Tier 1 among non-exact 4.2% (17/409).
All 11 exact and all 17 Tier 1 are Hindi; Bengali uses the `en` bundle
and is 0 exact. The old n=10 Table 1 paddle row is superseded.

Along the way the project found and fixed its own measurement bugs —
which is part of the science, not housekeeping. Whitespace-only
differences were being scored as letter substitutions until Tier 0 was
applied inside the assist heuristic. Pipe-as-danda and space-before-
punctuation cases looked like “genuine misreads” until Tier 1 grew to
cover them. Anusvara pairs were first misclassified as Tier 2 until it
became clear they are a deterministic sandhi rule and belong in Tier 1.

### What purpose this serves in everything that follows

Stage 0 is not a side quest. Every later accuracy number in this project
— instrument training curves, Probe 5 calibration, Probe 6 synthetic-
vs-real gap — is supposed to use the same notion of “correct”: grapheme-
aware, Tier-1-aware, Tier-2-aware where appropriate. If you skip Stage
0, you will later congratulate a model for “improving” when it only
learned to emit a different valid spelling.

> **What to remember.** Before you can improve Indic OCR, you have to
> stop calling valid alternate encodings “errors” — and that fraction
> is large enough to change how you read any published score.

---

## Chapter 2 — Turning Text Back Into Pixels, On Purpose

Chapter 2 is about building a controlled **text → image** machine so
that you can deliberately decide what the OCR model gets to see during
training.

> **Link to first principles:** the exposure **dial** (natural / flat /
> inverted; TV natural **0**, flat ≈ **0.047**, inv ≈ **0.005**) exists
> so we can ask Question A — but Probe 1 showed flattened/inverted models
> never learned to read; mechanistic claims use **natural** checkpoints only
> (Decision #66).

That sentence only makes sense once you start from the research
question. So that is where we begin.

### What are we actually trying to find out?

The project wants to answer:

> If an OCR model performs badly on a particular Indic character, is
> it because that character is inherently difficult to recognize, or
> simply because the model did not see it often enough during
> training?

Take two grapheme clusters as a toy example:

- `क` appears 10,000 times in training
- `ज्ञ` appears 100 times

Suppose the trained model then gets 95% accuracy on `क` and 60% on
`ज्ञ`. You **cannot** immediately conclude that `ज्ञ` is visually
harder. Maybe the model simply needed more exposure.

So we need an experiment where **we control exposure ourselves**.

That is the entire scientific reason Stage 1 exists. The renderer is
not a pretty picture generator. It is experimental apparatus.

### The basic experiment: three ways to teach the same model

Imagine you are teaching a child to recognize letters. You can give
them data in three different policies.

**Natural.** Give them data the way language usually arrives: `क`
appears a lot, `ज्ञ` appears rarely, `म` appears a lot, and so on.
This is the corpus as it is.

**Flattened.** Deliberately balance the training data so that common
and rare clusters get closer to the same number of examples. The
child no longer mostly practices the easy, frequent letters.

**Inverted.** Deliberately give rare characters *more* exposure and
common characters *less*. The formerly starved `ज्ञ` becomes common;
the formerly common `क` becomes scarce.

Then you train the **same model architecture** under all three
conditions, with total data volume held fixed (or explicitly matched).
Now you can ask a causal question:

> When I changed only how often each grapheme appeared, how did
> recognition change?

That is Probe 1’s experiment. Chapter 2 is the machinery that makes
the three training worlds exist as images, not as wishful thinking
about text files.

### But the model needs images, not text

This is the key reason the chapter is titled the way it is.

The model we train is an OCR model. It does not receive:

```text
ज्ञानी
```

as input. It receives an *image* of that word (or of a whole line /
page containing it), and has to produce the string `ज्ञानी`.

So if we want to control the training distribution, we need a way to
take carefully controlled text and turn it into realistic-looking
document images. That converter is the **renderer**:

```text
Controlled text
      ↓
   Renderer
      ↓
 Image of a document
      ↓
   OCR model
      ↓
 Predicted text
```

In code, that pipeline is centered on `src/renderer/render.py`, fed by
`glyph_frequency.py` (the dial), `layout_sources.py` (where text sits
on the page), and `degradation_profile.py` (how damaged the page looks).

### Why can’t we just use real scanned documents?

Real documents are essential later — as a reality check, and as the
source of measured blur and noise. But they do not give us enough
*control* for the causal experiment.

Suppose you download a thousand Hindi documents and count graphemes.
You might find something like 50,000 occurrences of `क` and 200 of
`ज्ञ`. You do not get to say: “Actually, give me 10,000 examples of
`ज्ञ` while keeping everything else roughly the same.”

You could try selecting documents that happen to contain rare
conjuncts. But then other things change too: sentence content, fonts,
layouts, topics, image quality, word distributions. You would no
longer know **what caused** the model’s performance to change.

The renderer lets us create the experiment instead of hoping the web
already contains it.

### What the renderer actually does

In plain language, the renderer says: give me text, and I will turn it
into a realistic document image — and I will remember exactly what
text I placed there.

So from:

```text
यह एक उदाहरण है।
```

you get a page image, plus automatic ground truth such as:

```json
{
  "image_path": "page_001.png",
  "text": "यह एक उदाहरण है।"
}
```

That pair is what training and evaluation need. Without exact ground
truth, Probe 1 cannot attribute errors to exposure in the first place.

A verification run of 100 Tier A pages under
`data/cache/renders/verify_100/` shows mean render time about 82 ms
(max about 353 ms) — well under the Stage 1 acceptance bar of one
second per page.

### Why HarfBuzz matters

This part is easy to misunderstand. You cannot simply draw Indic text
character-by-character the way a naïve Latin blit works.

Take `कि`. The underlying Unicode representation contains multiple
pieces, but visually they must be positioned together correctly. Even
more complicated are conjuncts such as `ज्ञ` or `क्ष`. Those are not
“letter plus letter” in pixels; they are shaped glyphs.

So the renderer needs a **text shaping engine**. That is what
**HarfBuzz** does:

```text
Unicode code points
        ↓
     HarfBuzz
        ↓
correct glyph shapes + positions
        ↓
       pixels
```

This project uses `uharfbuzz` for advances and cluster IDs (so each
grapheme cluster gets a bounding box) and Pillow with raqm/HarfBuzz
under the hood for painting, so the pixels and the boxes agree
(Decision #27). The synthetic image must look like real Indic writing;
otherwise you are training the OCR model on fake visual patterns and
Probe 1 measures the wrong thing.

### What “grapheme cluster” means here

Chapter 0 and Chapter 1 already introduced this, but it becomes
operational in the renderer.

A Unicode *code point* is an individual encoded piece. A *grapheme
cluster* is closer to “one thing a reader visually perceives as a
unit” — often a consonant plus vowel signs plus other marks.

The project does not want to say: “Show the model this Unicode code
point 5,000 times.” It wants to reason: “Show the model this **visual
grapheme unit** 5,000 times.” That is why
`src/renderer/glyph_frequency.py` counts and targets frequencies with
Unicode grapheme segmentation (`regex` `\X`), and only for clusters
that contain Indic script characters — so the dial cannot waste itself
promoting a stray `%` or Latin digit (Decision #25).

### Then there is the layout problem

A document is not text floating on a white background. Real documents
have margins, columns, headers, tables, forms, different text
positions, different fonts.

If the model only ever trains on:

> white background + one centered sentence

then any finding about “exposure” is entangled with “the model has
never seen a real page.”

So the project takes **layout geometry from real documents**.
Conceptually:

```text
Real document
      ↓
extract geometry (columns, headers, margins, …)
      ↓
reuse that geometry
      ↓
pour our controlled Indic text into those regions
```

`layout_sources.py` pulls from Internet Archive scans (via IIIF page
images), government PDFs, and Wikipedia Indic articles as printable
PDFs, and caches templates in `data/cache/layouts/bank.json`
(Decision #9). Born-digital PDFs donate their text and table boxes
directly. Camera scans get columns inferred from ink projection
profiles — an old, deterministic trick, used here because Stage 1 must
not introduce a second neural net whose mistakes would leak into Probe
1 (Decision #21).

**Honest gap:** in this checkout the bank has 28 templates (mostly
single-column, a couple of two-column, one marginalia). It does not
yet hold real `form` or `table-embedded` pages. That gap blocks Stage
3’s reading-order-vs-complexity curve later. It does **not** block
Probe 1’s glyph-frequency experiment, which wants layout held as fixed
as possible while only exposure moves.

### Why degradation?

Clean synthetic images are too easy. Real documents have blur, noise,
skew, show-through, scanning artifacts.

Instead of arbitrarily saying “add Gaussian blur with radius 1.2,” the
project measures those properties from real scans and stores them as
an empirical joint distribution (`degradation_profile.py`). Sampling
draws a whole page’s four-tuple — blur, noise, skew, show-through —
so the damage stays correlated the way it was on paper (Decision #24).

```text
Real scanned documents
        ↓
measure blur / noise / skew / show-through
        ↓
empirical distribution
        ↓
sample realistic degradation
        ↓
apply to a clean synthetic page
```

**Measured** on the fitted profile in this checkout (n=22 pages): blur
median about 1.16, noise median about 1.35, skew 90th percentile about
5.6°, show-through median about 0.044. The first calibration attempt
used line-level synthetic images and reported every real book page as
“perfectly sharp” — a scale mismatch, not a scientific result. The fix
was to calibrate blur against a full Wikipedia-print page rasterized
at a matching resolution.

### The three tiers are three levels of realism

The same renderer serves three jobs. Confusing them is how people
accidentally destroy the experiment.

**Tier A — controlled experiment.** Fixed font, fixed (usually zero)
degradation, and **only** glyph-frequency mode changes. This is the
important one for Probe 1. If accuracy changes across natural /
flattened / inverted, you want to be able to say: the main thing I
changed was exposure.

**Tier B — more realistic.** Now fonts, layouts, and degradation can
vary by sampling from the measured pools. This asks: does the finding
survive when images get messier?

**Tier C — real documents.** No synthetic rendering. Use actual scans
plus existing transcriptions. This is the reality check (Probe 6’s
world). Cluster boxes are not invented here; the question is text-level
behavior on real paper.

Conceptually:

```text
Tier A  →  Does the controlled exposure experiment work?
Tier B  →  Does it survive realistic variation?
Tier C  →  Does it hold on real documents?
```

### The clever part: natural / flattened / inverted

This is the most important mechanism in Chapter 2.

Suppose a natural corpus looks roughly like:

| Grapheme | Natural count (toy) |
|---|---:|
| `क` | 10,000 |
| `म` | 8,000 |
| `त` | 7,000 |
| `ज्ञ` | 500 |
| rare conjunct | 100 |

**Natural** keeps approximately that shape.  
**Flattened** pushes toward roughly equal mass across the observed
Indic support.  
**Inverted** swaps rank: rare things inherit the mass of common ones.

The **total amount of training data stays matched**, but **which
graphemes receive that exposure changes**. That is what makes it an
experiment rather than “just train on more data.”

In code, `glyph_frequency.resample_corpus()` implements the dial.
Stage 1’s acceptance gate is explicit: realized frequencies must match
the target within total-variation distance **TV ≤ 0.08**
(`TARGET_TV_TOLERANCE`).

**Measured** on the 60-line Hindi ground-truth slice with seed 0:

| mode | TV to target | within 0.08? |
|---|---:|---|
| natural | 0.000 | yes |
| flattened | ≈ 0.047 | yes |
| inverted | ≈ 0.005 | yes |

### Why couldn’t they just select sentences?

This is one of the subtler points, and it is why the first algorithm
failed.

Imagine these sentences:

```text
Sentence A: क क क म
Sentence B: क म त
Sentence C: ज्ञ क्ष
```

You might think: “I’ll just pick more sentences containing `ज्ञ`.”
But every sentence contains **multiple** graphemes. Choosing Sentence
C to increase `ज्ञ` also increases `क्ष`, and potentially many other
characters. So by only selecting existing sentences, there are hard
limits to which frequency distributions you can create. The achievable
histograms live inside the convex hull of sentence bags — and that
hull does not reach “uniform” or “inverted” on real Indic text.

On the Hindi GlotOCR slice, a greedy oracle that only picked existing
sentences could not get below roughly TV 0.29 (flattened) / 0.73
(inverted). So the project switched to **synthesis** (Decision #29):

> I know the exact number of each grapheme I want. Now construct text
> that satisfies that quota.

It allocates an exact integer multiset from the target distribution
(largest-remainder), then packs those glyphs into sentence-shaped
strings with a bigram walker trained on the source corpus. The result
is only as language-like as the bigram table allows. That
“naturalness confound” is why Probe 3 (blank / noise images) exists
later: to measure how much apparent reading is actually language-model
guessing (Decision #10).

### What the manifests are — and what they are not

After rendering pages, the project does not train the instrument on
full pages first. It creates **line crops**.

A page becomes several training rows:

```text
page.png
   ↓
 line 1 image  +  "यह एक उदाहरण है।"
 line 2 image  +  "भारत में कई भाषाएँ हैं।"
 ...
```

written as JSONL:

```json
{"image_path": ".../line_001.png", "text": "यह एक उदाहरण है।"}
{"image_path": ".../line_002.png", "text": "भारत में कई भाषाएँ हैं।"}
```

That is what lives under `data/manifests/`:

```text
hindi_natural.jsonl / hindi_flattened.jsonl / hindi_inverted.jsonl
bengali_natural.jsonl / bengali_flattened.jsonl / bengali_inverted.jsonl
```

Those files are **training instructions**: here is an image of a line;
here is the correct text. They are produced by
`export_line_manifest.py` and the batch driver
`export_manifest_scaled.py` (canonical line height 70 px — five ViT
patches of 14 px).

This also clarifies a common confusion. Seeing a Colab log like
`[bengali/inverted] page 100/100` means **Stage 1 data preparation**
(resample → render → crop → append JSONL). That can finish relatively
quickly. It is **not** the same as Stage 2 neural training, which loads
those images thousands of times through an encoder–decoder and updates
tens of millions of parameters. Manifest generation is the setup for
the experiment; training is the experiment.

### How Chapter 2 sits in the whole scientific chain

Putting the pieces together:

```text
           Real documents
                 │
                 ▼
      learn layouts + degradation
                 │
                 ▼
            RENDERER
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  Natural    Flattened    Inverted
     │           │           │
     └───────────┼───────────┘
                 ▼
            line images
                 │
                 ▼
        train from scratch
                 │
                 ▼
           OCR instrument
                 │
                 ▼
              PROBES
                 │
                 ▼
   Was this glyph hard because it is
   visually hard — or because the model
   rarely saw it?
```

If someone asks what Chapter 2 is for, in one sentence:

> We are building the infrastructure that turns controlled Indic text
> into realistic document images with precisely set grapheme-frequency
> distributions — so that later, when we train a model from scratch
> under natural / flattened / inverted conditions, we can ask whether
> errors come from lack of exposure or from intrinsic visual
> difficulty.

Without this chapter’s machine, Probe 1 has nowhere to plug in. With
it, exposure stops being an accident of the web and becomes a dial you
can turn.

> **What to remember.** If you cannot set exposure on purpose, you
> cannot claim to have separated “rarely seen” from “hard to see.”

---

## Chapter 3 — Learning to Read From Nothing (the Instrument)

Chapter 2 built the **controlled training data**. Chapter 3 builds the
**OCR model that will actually learn from it** — and that we can open
up later as a measurement instrument for Question B (does confidence
track visual evidence?). Architecture and training numbers live in
[`docs/training_config.md`](./docs/training_config.md) (**19,607,104**
params); the model is undertrained on purpose — accuracy is not the
deliverable. Read **First principles** before this chapter if you want
the position-0 dissociation first.

The key idea is:

> We deliberately start with a model that knows nothing about Indic
> scripts, so that later we can ask (A) whether performance depends on
> how much exposure each grapheme received, and (B) whether its
> confidence tracks the image or only language plausibility.

### Why can’t we just use an existing OCR or VLM?

Suppose you take a pretrained vision–language model and fine-tune it
on natural Bengali, flattened Bengali, and inverted Bengali. You might
see: “The inverted model performs better on rare graphemes.”

There is a huge problem. The pretrained model may have **already
learned Bengali** before your experiment started:

```text
Pretraining
   ↓
Model already knows some Bengali
   ↓
Your natural / flat / inverted training
   ↓
Observed performance
```

You cannot tell how much of the final behavior came from your
controlled exposure versus the model’s previous exposure. Your
fine-tuning mixture is a rounding error against that history, and the
causal claim collapses on the first serious question.

That is why Decision #1 is non-negotiable: **the instrument starts
blank.**

```text
Randomly initialized model
          ↓
Natural training ──────┐
Flattened training ────┼──→ Compare
Inverted training ─────┘
```

Now exposure is controlled from the beginning. A second,
production-shaped model (the **demo**, Chapter 4) can come later for
architecture proof. They must not be the same weights doing double
duty.

### What exactly is this “instrument”?

Think of it as a **scientific measuring device**, not a production OCR
system.

You are not trying to build “the world’s best OCR model.” You are
trying to build “a model simple enough that I can inspect what it
believes and why.” That is why it is called an instrument.

Later you want to ask:

- What probability did the model give to `ज्ञ`?
- What were its next-best guesses?
- How confident was it on a blank image?
- Does confidence correspond to correctness?
- Did increasing exposure to `ज्ञ` improve its recognition?

A closed commercial OCR API generally will not let you inspect all of
that. The instrument’s job is to make Probes 1–5 possible.

### The model has two major parts

The architecture is a small encoder–decoder:

```text
IMAGE
  │
  ▼
┌──────────────┐
│   ENCODER    │  vision model
└──────────────┘
  │ visual features
  ▼
┌──────────────┐
│   DECODER    │  text model
└──────────────┘
  │
  ▼
TEXT
```

Very roughly: the **encoder looks at the image**; the **decoder
generates the text**. The code lives under
`src/models/instrument/` — `encoder.py`, `decoder.py`, `tokenizer.py`,
`train.py`, `generate.py`.

### Encoder: looking at the image

The encoder is a small Vision Transformer (`InstrumentEncoder`).

The image is divided into **14×14 pixel patches**. Each patch becomes
a vector. The Transformer then lets patches interact with each other —
so rather than saying “this individual patch is the letter,” the model
can learn relationships such as “this mark above the base character
belongs to that character.”

This encoder has:

- 6 Transformer layers
- hidden dimension 320
- 14×14 image patches
- sinusoidal positional information
- grayscale line images as input

**Measured** encoder size: **7,460,800 parameters**. That number does
not depend on vocabulary size; it is the vision tower alone.

### Decoder: turning vision into text

The decoder (`InstrumentDecoder`) receives the visual information from
the encoder and generates the transcription one unit at a time.

Suppose the image says `भारत`. The decoder might generate:

```text
<BOS>
  ↓
भ
  ↓
ा
  ↓
र
  ↓
त
  ↓
<EOS>
```

At every step it asks: given the image and everything I have generated
so far, what should come next? That is **autoregressive generation**.

In this codebase the decoder has five layers and hidden size 384
(Decision #36). A linear projection inside `InstrumentModel`
(`train.py`) bridges the encoder’s 320-d memory to the decoder’s 384-d
width, so each half stays independently smoke-testable.

**Measured** at Hindi natural vocabulary size $|V|=367$ (tied embedding
and output head): the full `InstrumentModel` is **19,607,104**
parameters. The encoder alone is 7,460,800. Because the output matrix
is tied, growing the grapheme table from a smoke vocabulary to
Devanagari barely moves the total; the old 30–60M envelope was a
design ceiling, not the trained Hindi checkpoint (Decision #65).

Probe 1 still trains **nine** Hindi runs (natural / flattened /
inverted $\times$ three seeds). The paper’s mechanistic measurements
use only the three natural seeds (Decision #66): flattened and inverted
collapsed to near-zero line accuracy, so they are not extra copies of
the same reader.

### What “causal mask” means

The decoder must not cheat.

Suppose the correct sequence is `भ → ा → र → त`. When predicting
`र`, it may see `भ, ा`, but it must not be allowed to see the future
token `त`. The causal mask enforces: current prediction may attend to
**past tokens only**. That makes training resemble actual generation.

### What “cross-attention” means

The decoder needs two kinds of information:

- **What have I already generated?** — handled by **self-attention**
  (with the causal mask).
- **What is actually in the image?** — handled by **cross-attention**
  to the encoder’s visual features.

Conceptually:

```text
                IMAGE
                  │
                  ▼
             ENCODER
                  │
           visual features
                  │
                  ▼
            CROSS-ATTENTION
                  ▲
                  │
        previous generated text
                  │
                  ▼
              DECODER
                  │
                  ▼
             next grapheme
```

When the decoder decides what comes next, it can look back at the
image. Without that, it would be a language model guessing from prior
tokens alone — which is exactly the failure mode Probe 3 is designed
to catch.

### Why the tokenizer is unusual

This is one of the most important design decisions in the whole
project (Decision #2).

A normal language model might use **BPE**, which turns text into
pieces driven partly by text frequency. But the research question is:
how does *visual* recognition depend on exposure to individual
grapheme clusters?

If the tokenizer itself merges frequent things differently, you have
introduced another variable. So the instrument uses:

> one grapheme cluster = one token

That gives Probe 1 a clean relationship:

```text
exposure  ↔  grapheme  ↔  recognition
```

instead of:

```text
exposure  ↔  BPE segmentation  ↔  token frequency  ↔  recognition
```

The vocabulary is built fresh per training run from that run’s corpus
(`GraphemeTokenizer` in `tokenizer.py`), with fixed special tokens
`<PAD>`, `<BOS>`, `<EOS>`, `<RARE>`. Clusters below a frequency floor
map to `<RARE>` at encode time.

### Why `\X`?

The tokenizer uses the `regex` module’s `\X` pattern — Unicode
**grapheme cluster** boundaries (Decision #7). That is the same unit
Chapter 2’s frequency dial controls.

So Chapters 2 and 3 connect on purpose:

```text
Chapter 2
"What visual units do we control exposure for?"
              ↓
       Grapheme clusters
              ↓
Chapter 3
"What units does the model predict?"
              ↓
       Grapheme tokens
```

If those units disagreed, Probe 1 would be measuring two different
notions of “how often” at once.

### Why train on lines instead of pages?

The renderer creates full pages, but the instrument trains on **line
crops** (Decision #37).

Why?

- **Memory.** Full pages mean many more ViT patches → more compute and
  GPU memory.
- **Speed.** Line-level training lets you run many more updates.
- **Simplicity.** The research question here is primarily about
  **recognition**, not page layout. Layout complexity is a later
  chapter’s problem.

So the training interface is deliberately clean:

```json
{"image_path": ".../line_001.png", "text": "यह एक उदाहरण है।"}
```

That is why the manifests from Chapter 2 matter. Stage 1 and Stage 2a
meet at JSONL rows, not at ad-hoc tensors. Canonical line height is 70
pixels — five ViT patches of 14 px.

### What is teacher forcing?

During training, suppose the correct answer is `भारत`. The model
predicts one token at a time. With **teacher forcing**, when training
the next prediction we feed it the **correct previous token**, not its
own potentially wrong previous guess:

```text
Image + <BOS>           → predict भ
Image + <BOS> भ         → predict ा
Image + <BOS> भ ा       → predict र
...
```

That makes supervised training much more stable. During actual
generation (`generate.py`), however, the model must use **its own**
previous predictions — the distribution shift between those two modes
is a known property of autoregressive training, not a bug unique to
this repo.

### Why fp16?

Heavy training is designed for a free Colab **T4**. The project uses
**FP16** (16-bit floating point) to reduce memory and speed training.
Turing GPUs like the T4 do not support BF16 the way newer cards do, so
the training code is built around FP16 and gradient checkpointing —
not because FP16 is theoretically special, but because that is the
hardware constraint the whole project accepted.

### Why checkpoints are important

Training can take hours. You do not want five epochs of progress to
vanish when Colab disconnects.

Instead:

```text
train → checkpoint → train → checkpoint → …
Colab dies → restart → load latest checkpoint → continue
```

Resumable checkpoints are a hard engineering requirement in this repo
(see `AGENTS.md` and `train.py`), not a nice-to-have. The same rule
applies to every long batch script: progress per item, resume by
default.

### What `generate.py` does — and why “open” matters

After training, you give the model an image. It greedily generates
tokens until `<EOS>` (Decision #38: greedy only, no beam search, no KV
cache at this scale).

But `generate.py` does not only return `"भारत"`. It also returns:

- per-step **confidence** (max softmax probability at each step)
- **top-k alternatives** at each step (what the model almost said)

That information is extremely important for the probes:

| Probe | Needs from generation |
|---|---|
| Probe 2 (confusion) | top-k alternatives — what it almost predicted |
| Probe 3 (blank control) | text + confidence on empty / noise images |
| Probe 5 (calibration) | confidence vs actual correctness |

Imagine the model predicts `ज्ञ` with confidence 0.61, and the
alternatives are `ग` 0.22, `ज` 0.10, `क्ष` 0.04. Now you can study
**what the model almost thought**. A closed API that only returns the
string `ज्ञ` cannot support that experiment.

### Why the model has to be small

A huge pretrained model might be much better at OCR. It would be a
terrible **scientific instrument** for this particular question,
because it has too much hidden history.

The project deliberately chooses:

> small + from scratch + inspectable

over:

> large + pretrained + powerful

because the goal is not maximum OCR accuracy. It is trying to isolate
**causal relationships**.

### How Chapter 3 connects everything so far

```text
CHAPTER 1
Define what "correct" means
        │
        ▼
CHAPTER 2
Create controlled image data
        │
        ├── Natural
        ├── Flattened
        └── Inverted
        │
        ▼
CHAPTER 3
Train a blank OCR instrument
        │
        ├── Encoder → sees image
        └── Decoder → generates graphemes
        │
        ▼
Later probe chapters
        │
        ├── Does exposure matter?
        ├── What does it almost predict?
        ├── Is it actually looking at the image?
        └── Does confidence mean correctness?
```

### The most important distinction: instrument vs demo

There are **two different models** in the overall project. Do not mix
their purposes.

**Instrument** — from scratch, no Indic pretraining, controlled
experiments, scientific probes. Asks: *why does OCR behave this way?*

**Demo** (Chapter 4) — pretrained VLM, LoRA / fine-tuning,
production-style system. Asks: *can we adapt a modern model into a
useful OCR system?*

### What is and isn’t evidenced in this checkout

The code path is real. `make smoke-test` runs tokenizer, encoder,
decoder, nine fake Probe 1 training runs, and generation end to end
with no GPU and no `data/raw` — an architecture proof that produces
**zero scientific findings by design**, and it currently passes.

Real Hindi / Bengali training checkpoints and probe result JSONL files
are not necessarily present in a fresh checkout (they often live on
Drive / Colab). README and the site report early Probe 3/5 numbers
from one checkpoint; until those artifacts are re-run here, treat them
as **reported**, not as something this book independently verified.

Without the instrument, Probe 2 is impossible against a closed API,
Probes 3 and 5 become anecdotes, and Chapter 2’s dial has nowhere to
plug in. The instrument is the reason Stages 0 and 1 were worth
building carefully: they feed a model you can actually interrogate.

> **What to remember.** The instrument is small and blank on purpose —
> so that “what did the model see?” and “what did it believe?” are
> questions you can answer, not stories you tell about someone else’s
> API.

---

## Chapter 4 — Teaching an Existing Model a New Trick Cheaply (the Demo)

This chapter is basically saying:

> We actually have two different goals, so we need two different models.

Chapter 3 already built one of them — the instrument. Chapter 4 exists
to explain the **other** goal, and why merging the two goals into one
set of weights would destroy the science.

### The first model = the Instrument

This is the model from Chapter 3. Its purpose is **research**:

> If I change how often the model sees certain Indic graphemes, what
> happens?

For that question, we need a model that starts with **zero prior
knowledge of Indic text**.

Otherwise, suppose we take a pretrained VLM that has already seen
millions of Hindi / Bengali examples and then train it more heavily on
rare Hindi characters. If performance improves, we cannot confidently
say: “It improved because we increased exposure to those characters.”
The model may already have learned them during pretraining.

So the instrument is deliberately:

```text
blank → controlled training exposure → measure behavior
```

### The second model = the Demo

The demo has a completely different purpose.

Instead of asking “what caused the model to learn this?”, we are asking:

> Can we build something resembling a practical / production OCR–VLM
> system?

A real-world system usually **does not start from random weights**.
Training a large VLM completely from scratch is enormously expensive.
Instead the usual path is:

```text
Pretrained model → add LoRA → fine-tune on our OCR data
```

That is the demo’s world. Decision #1 records this split explicitly:
one model for causal probes, one model for architecture demonstration.
They must not be the same weights doing double duty.

### What is LoRA?

Imagine a pretrained model has a billion parameters. You do not want to
modify all of them just to teach it your OCR task.

**LoRA** (Low-Rank Adaptation) essentially says: keep the original model
frozen, and learn a relatively small set of additional parameters that
steer it toward our task.

```text
                 PRETRAINED MODEL
                /               \
          frozen weights      LoRA adapters
             ↓                    ↓
        existing knowledge    OCR-specific adjustment
                \               /
                 → OCR output
```

The important advantage is that **you train far fewer parameters**,
which makes adaptation much cheaper — cheap enough, in principle, to
attempt on a free Colab T4 once the base model fits in memory.

That is the standard modern pattern for “teach an existing model a new
trick without rewriting every weight.” It is also exactly why the demo
cannot answer Probe 1’s causal question: the frozen weights already
contain someone else’s exposure history.

### Why can’t the Demo replace the Instrument?

This is the most important distinction in the chapter.

Imagine we train:

```text
Pretrained VLM
      ↓
LoRA
      ↓
Hindi training (natural / flat / inverted)
      ↓
measure effect of glyph frequency
```

We might observe: rare-glyph performance improved by 8%. But what
caused that?

Possibilities include:

- the new exposure you carefully set,
- knowledge already present in the pretrained model,
- interactions with its existing language knowledge,
- its existing visual knowledge,
- or the LoRA adaptation itself.

We **cannot isolate exposure cleanly**. That is why the two machines
answer different questions:

**Instrument**

```text
Random initialization
        ↓
Controlled exposure
        ↓
Scientific measurement
```

**Demo**

```text
Pretrained model
        ↓
LoRA adaptation
        ↓
Practical OCR system
```

Instrument = *why does OCR behave this way?*  
Demo = *can we adapt a modern model into a useful OCR system?*

### What “production-shaped” means here

The demo is not just `image → text`.

The project wants to demonstrate the kind of **decomposition** a real
document-processing system might use — closer to the shape of
Sarvam-style digitisation pipelines than to a single monolith:

```text
                 Document image
                       │
                       ▼
              ┌─────────────────┐
              │ Layout detection│
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Text block          Table / form
             │                   │
             ▼                   ▼
       Reading order       Structure handling
             │
             ▼
        OCR / VLM
             │
             ▼
       Structured output
```

So the demo is intended to show that you understand **system
architecture** — layout, reading order, recognition as separate
concerns — not merely how to train an OCR model end to end. That is
architecture demonstration, not exposure science. Chapters 5 and 6
teach the reading-order and RL ideas that would eventually hang off
this shape; they are not claimed as finished in this phase.

### Why the Demo is currently deferred

The project has limited compute and time, particularly because heavy
work is designed around a **T4 / Colab** environment.

The demo involves several additional components:

- choosing a pretrained VLM
- LoRA fine-tuning
- supervised training
- potentially RLVR later (Chapter 6)
- layout detection
- reading-order handling
- structured / table handling

That is a substantially larger engineering project than “train a blank
instrument three ways.” So the current priority stays:

```text
Stage 0  → error taxonomy
Stage 1  → controlled renderer
Stage 2a → from-scratch instrument
Probes   → scientific findings
```

The demo comes separately. It is **not** gated on the instrument’s
findings; it is gated on time and on T4 memory headroom. This chapter
is therefore not a claim that the demo is finished. It is the reason
the demo must stay **separate** from the instrument: if you use a
pretrained backbone for Probe 1, you are no longer controlling
exposure.

### What `benchmark_base_models.py` is doing

Before choosing the model for the demo, the repo wants an honest answer
to:

> Which pretrained model can actually fit our LoRA experiment on the
> available GPU?

The two candidates named in Decision #3 are:

- **SmolDocling-256M** (`ds4sd/SmolDocling-256M-preview`)
- **LightOnOCR-1B** (`lightonai/LightOnOCR-1B-1025`)

`src/models/demo/benchmark_base_models.py` is supposed to measure
things such as **peak VRAM usage under LoRA** (and, in `--inspect`
mode, the real attention module names you must pass to LoRA — do not
guess those). So instead of arbitrarily saying “let’s use the 1B
model,” you first measure:

```text
Model A → peak VRAM → fits / doesn’t fit
Model B → peak VRAM → fits / doesn’t fit
```

Then choose based on actual hardware constraints, including headroom
for layout and reading-order modules that would also be resident later.

That measurement **ran on Colab T4**. Decision #84 records the close:
SmolDocling-256M-preview at 1.63 GB dummy-LoRA peak. Newer successors
(granite-docling-258M, LightOnOCR-2) were in the same pass; they were
not silent swaps for the Decision #3 pair.

### The key distinction to remember

If someone asks “why do you have two models?”, a strong answer is:

> Because they serve different scientific and engineering purposes.
> The instrument is trained from scratch so I can control the model’s
> prior exposure and study questions like exposure versus intrinsic
> script complexity. The demo is a separate pretrained VLM adapted with
> LoRA, because that is closer to how we would actually build a capable
> production system. Using the pretrained model for the causal
> experiment would confound the exposure variable with what it already
> learned during pretraining.

And the one-line version:

> **Instrument = scientific measurement. Demo = practical system
> demonstration.**

That separation is one of the central design decisions of the entire
project (Decision #1).

> **What to remember.** Fine-tuning a pretrained model can prove you can
> ship a shape; it cannot prove what exposure caused, because exposure
> already happened before you arrived.

---

## Chapter 5 — Where Does the Text Go on the Page

### The problem reading order actually is

Human readers do not always go top-to-bottom, left-to-right. A
two-column newspaper page is read down the left column, then down the
right. Marginalia interrupts. Tables have an internal grammar. If an OCR
system emits the right words in the wrong order, character accuracy can
look fine while the document is unusable for search, quoting, or
retrieval.

So reading order is not a classification problem (“is this block a
title?”). It is a **permutation problem**: given the set of blocks, what
sequence should they be read in?

### Why Kendall tau, not accuracy

Accuracy on order would treat “swapped two adjacent blocks” the same as
“read the page backwards.” Both are “wrong,” but they are not equally
wrong. **Kendall tau** measures how many pairwise orderings agree
between the predicted sequence and the ground-truth sequence. It gives
you a graded sense of how far off the ordering is.

This project wants that score **as a curve against layout complexity**:
single-column → two-column → marginalia → table-embedded. One average
number would hide whether the system only fails when the page gets hard.

### Tables, scoped down on purpose

An earlier idea was “table to prose”: generate a sentence per row. That
needs its own evaluation methodology (what counts as a correct
paragraph?) that this project does not have budget for. Decision #12
cuts it down to something checkable: after OCR, **is each cell still
bound to the correct column header?** The renderer can provide clean
ground truth for that. It also matches what you would compare against
Sarvam’s Extract-style structured output later.

### How this project would actually build it

Two blueprints, one for ordering, one for tables, both scoped to what
is actually measurable rather than what sounds impressive.

**Reading order: pairwise relation, not a single global decision**

The naive framing — "predict the correct order of N blocks" — is a
much harder learning problem than it needs to be, because it asks a
model to reason about a whole page at once. A more tractable framing
mirrors how Kendall tau itself is computed: for every *pair* of
blocks, ask one small, local question — **does block A come before
block B?** — using only their relative position, size, and maybe a
little of their recognized text as features.

This has a genuinely nice property: the training signal and the
evaluation metric are now the same shape. Kendall tau counts pairwise
agreements; the model is trained to predict pairwise agreements. There
is no mismatch between what the model optimizes and what gets
reported.

At inference, you have a pile of pairwise "A before B" predictions for
every pair on the page, and you need one global order out of them.
Two ways to do that:

- **Count and sort.** For each block, count how many other blocks it
  was predicted to come before. Sort blocks by that count, descending.
  Simple, forgiving of a few wrong pairwise calls, and cheap to
  implement.
- **Pointer network.** A sequence decoder that, at each step, looks at
  which blocks remain and "points to" the next one to read — the same
  mechanism (Vinyals et al., 2015) used elsewhere for problems where
  the output is a permutation of the input. This guarantees a valid
  ordering by construction (no ties, no cycles), but it is a harder
  model to train well on a small blueprint dataset.

Given this project's actual constraint — a free T4 and a still-partial
layout bank — the pairwise classifier is the more buildable first
version: fewer parameters, a training signal that matches the metric
directly, and a graceful failure mode (a few wrong pairwise votes just
nudge the sort order, they don't break everything). The pointer
network stays the documented alternative for later, exactly as
`IMPLEMENTATION.md` already lists both options.

**Table binding: your header-matching idea is table structure
recognition, formalized**

Restated precisely, the idea is: find the header row, then for every
data row, walk across it pairing each cell with the header above it,
and assemble a record — `{"name": "Priya", "age": "27", "city":
"Mumbai"}`. That is exactly right, and it is a real, named task in the
literature: **table structure recognition**, specifically the
cell-to-header binding step of it. Four concrete stages:

1. **Header detection.** Which row is the header? Often just the first
   row, but real tables use bold text or shading instead — a small
   classifier over visual features (position, boldness) generalizes
   better than "always row 0."
2. **Column alignment.** For each cell below the header, which header
   does it belong under? On a clean, unrotated grid this is pure
   geometry — compare the cell's horizontal span against each header
   cell's span, take the best overlap. Real scanned tables skew and
   merge cells, which is exactly why this step needs to be *learned*
   rather than assumed once real data enters the picture, not just
   solved on the renderer's clean synthetic grid.
3. **Row grouping.** Cluster cells by vertical position into rows,
   same geometry-first logic, same caveat about skew.
4. **Assembly.** Walk each row in column order, pair every cell with
   its bound header, emit the record — this is the step that produces
   exactly the JSON-shaped output you described.

This four-step breakdown is *also* precisely what `table_binding.py`'s
metric (Decision #12) checks: not "did you reconstruct a paragraph,"
but "after steps 2 and 3, is each cell still correctly bound to its
column header?" Your intuition and the project's existing scoped-down
metric are the same idea, described two different ways — the metric
is just the part of your pipeline that's cheap to verify without also
having to solve full prose generation.

**Why this stays a blueprint, not a built module, in this phase**

Steps 2 and 3 need real, structurally hard tables — merged cells,
skew, multi-row headers — and the layout bank (Chapter 2) does not
yet hold those. Building the geometry-only version against clean
synthetic tables would look deceptively easy and tell you nothing
about the case that actually matters. The honest order to build this
in, when there's time: finish the layout bank's table-embedded
category first, then the geometry baseline, then the learned version
only where geometry demonstrably fails.

The metrics files (`src/eval/reading_order_metric.py`,
`src/eval/table_binding.py`) are now built and unit-tested. A geometric
y-then-x baseline on the live `bank.json` (28 templates) is in
`docs/tier2_stage3_reading_order.md`. You still cannot honestly report
a **demo-model** complexity curve until Stage 1 covers real forms and
table pages — empty `form` / `table-embedded` buckets are holes, not
zeros.

> **What to remember.** Reading order and table binding are the same
> kind of problem underneath — turning "a pile of correctly recognized
> text" into "the structure a reader actually needed" — and both are
> solvable as small, local, pairwise decisions rather than one big
> global guess.

---

## Chapter 6 — Reinforcement Learning, From Scratch

This chapter is about **changing how we train the model**.

So far, the project mainly uses **supervised learning**:

> Here is the image, and here is the correct text. Learn to produce
> this exact text.

Chapter 6 asks:

> What if instead we tell the model what makes an *entire* output good
> or bad, and let it learn from a reward?

That is **reinforcement learning (RL)**.

### Supervised learning

Suppose the image says `भारत एक देश है`. We give the model the image
paired with that exact string. During training the model produces a
sequence; we compare it against the ground truth and calculate a
**loss**. The model learns: my output should become more like the
ground-truth string.

That is very natural for OCR. Up to the demo’s supervised fine-tuning
stage, that is the default training story.

### But exact text isn’t the whole problem

Consider a document with a title, two columns, and a table. A model
could recognize every character correctly but still:

- put Column B before Column A
- mix up table cells
- skip a footnote
- omit an entire paragraph

So simply teaching “predict the next correct token” does not always
capture everything we care about. We want to evaluate the **whole
document output** — recognition *and* structure *and* whether anything
was quietly left out. That connects directly to Chapter 5’s reading-
order and table-binding concerns.

### Reinforcement learning changes the setup

Instead of comparing token-by-token against a single gold string during
every step of generation, RL does something closer to:

```text
Image
  ↓
Model
  ↓
complete output
  ↓
evaluate the whole reading
  ↓
REWARD
  ↓
model learns to maximize reward
```

One complete reading might score 0.92; another, with swapped cities or
missing rows, might score 0.65. Training pushes the model toward kinds
of outputs that receive higher rewards.

### What is RLVR?

**RLVR = Reinforcement Learning with Verifiable Rewards.**

The important word is **verifiable**. We are not asking a human “does
this OCR output look good?” We have information that lets us
**automatically** calculate whether the output is correct.

Because the synthetic renderer from Chapter 2 knows the ground truth —
correct text, correct reading order, correct table structure — we can
score a full reading without a human in the loop. That is why owning
the renderer is useful again here: controlled pages are not only
training fuel; they are a source of automatic, checkable rewards.

### The dangerous part: the model can game the reward

This is probably the most important concept in the chapter.

Suppose your reward is simply: how accurately did you recognize the
text you actually output? The model might discover a clever strategy:
**don’t output difficult text.**

Imagine the page says:

```text
This is an easy sentence.
This is a difficult sentence containing rare glyphs.
This is another easy sentence.
```

The model could output only the easy sentences. If your metric only
evaluates the text it produced, accuracy can look *higher* while the
OCR system has become worse. It has learned: if I’m uncertain, shut
up.

That is **reward hacking** / **reward gaming**. It is not a
hypothetical. It is the failure mode this project plans to demonstrate
on purpose.

### Coverage solves this particular problem

The planned reward is a sum with an explicit anti-omission term
(Decision #11):

```text
Reward =
    character accuracy
  + structure match (TEDS)
  + reading-order quality
  − coverage penalty
```

**Character accuracy** — did you recognize the text?  
**Structure match (TEDS)** — did you preserve table / document
structure? TEDS is a metric for whether a predicted table structure
matches the reference.  
**Reading-order quality** — did you put the blocks in the correct
order? (Chapter 5’s Kendall-tau world.)  
**Coverage** — did you actually cover the entire document? This is the
crucial anti-gaming term.

### A concrete example

Suppose the page contains 100 characters.

**Model A** reads all 100 but makes 10 mistakes → coverage 100%,
accuracy about 90%.  

**Model B** reads only the easiest 50 and gets them all right →
coverage 50%, accuracy 100%.

Without coverage, Model B looks better. With coverage, Model A gets
the higher reward and Model B is penalized for omission. That is why
the project explicitly includes coverage.

### The planned experiment is deliberately small

The project does not want dozens of RL experiments. It wants one
particularly informative comparison (Decision #11):

**Normal RLVR**

```text
accuracy + structure + reading order − coverage
```

versus **ablation** — remove coverage:

```text
accuracy + structure + reading order
```

Then ask: does the model start omitting difficult content? If it does,
you have demonstrated **why coverage matters**.

An **ablation** means: remove one component and see what changes. A
full sweep across every reward term would cost more than it teaches
here; the coverage-term removal is cheap (a single retrain) and points
at the mechanism rather than a spreadsheet of tiny deltas.

### Why the coverage ablation is still untrained

SFT for the demo now exists (100 LoRA steps on SmolDocling, Colab).
The coverage-term **retrain** still has not run. Cell 10 scored the
SFT adapter’s greedy decode with λ=0 and did not update the policy
(`docs/tier2_rlvr_ablation.md`). RLVR still sits behind the demo, not
the instrument:

```text
Instrument
   ↓
Understand OCR behavior (probes)
   ↓
Demo pretrained VLM + LoRA
   ↓
Supervised fine-tuning
   ↓
RLVR
```

The instrument is about scientific diagnosis. The demo + RLVR path is
about building a more production-like system and testing advanced
training methods. So this chapter is currently a **design /
blueprint**, not a completed experiment. When someone builds it, they
inherit the scientific bar: a verifiable reward, a known gaming mode,
and one ablation that shows the mechanism.

### How the chapters connect here

```text
Chapter 2 — CONTROL THE DATA
"What did we show the model?"
      ↓
Chapter 3 — BUILD AN INSTRUMENT
"What did the model learn?"
      ↓
Chapter 5 — CHECK DOCUMENT STRUCTURE
"Did it preserve order?"
      ↓
Chapter 6 — OPTIMIZE THE WHOLE OUTPUT
"Can we reward the behavior we actually want?"
```

And Chapter 6 exposes an important ML lesson:

> A model does not necessarily optimize what you intended. It
> optimizes the reward you gave it.

If the reward says that omitting difficult text is okay, the model can
learn to omit difficult text.

`src/models/demo/rlvr.py` implements that scalar (including λ=0). Unit
tests check the omission **shape**. A coverage-term-removed **retrain**
was not run. SFT exists (Colab Drive adapter); the trainer never
implements a policy update (`docs/tier2_rlvr_ablation.md`,
`docs/rlvr_scoping.md`).

> **What to remember.** RLVR lets us optimize OCR for whole-document
> quality, but the reward must include coverage — otherwise the model
> can improve its score by simply leaving difficult content out.

---

## Chapter 7 — What Does the Model Actually Know (the Probe Suite)

Chapter 7 is the payoff of everything in Chapters 0–6. If you read
**First principles: the central finding**, this chapter is the same story
with file paths, probe IDs, and alternative-explanation structure.

The easiest way to understand it is:

> We built a small OCR model not primarily to make it accurate, but so
> we can open it up and separate **correctness**, **confidence**, and
> **visual grounding** — Question B from the front of this book. The
> paper does **not** claim universal hallucination; it documents this
> failure mode in **this** instrument and tests weak transfer to Sarvam.

Think of the instrument as a patient in a lab, and the probes as
medical tests. Each probe also attacks a competing explanation for the
previous observation (undertraining → training curve; domain shift →
noise/scramble; “maybe the encoder still matters for confidence” →
ablation; “maybe only our toy model” → Stage 5–6).

### First: what are we trying to find out?

Suppose the model reads an image and outputs `भारत`, and the correct
answer is `भारत`. Accuracy = correct. That tells us almost nothing
about **how the model arrived there**.

We really want to know:

1. Does seeing a glyph more often make the model better at it?
   (Question A — exposure. Apparatus succeeded; causal β withheld.)
2. When it makes a mistake, what did it think the answer was?
3. Is it actually looking at the image, or just completing language?
4. When it says “I’m 99% confident,” is it actually right 99% of the
   time — and does that confidence mean the *image* supported the text?

Those questions (plus fairness of scoring, plus a reality check on
synthetic pages) are what the **probes** investigate. They are the
project’s real deliverable. The instrument exists so these questions
have somewhere to land.

### Probe 1 — Does exposure make you better?

This is probably the most important probe. It is the scientific
question from Chapter 2, now asked of a trained model:

> Is a glyph difficult because it is intrinsically difficult to
> recognize, or because the model simply didn’t see it enough?

Suppose `क` appears 10,000 times and `ज्ञ` appears 100 times, and the
model gets 95% on `क` and 50% on `ज्ञ`. You cannot immediately say
“`ज्ञ` is visually harder.” Maybe we simply didn’t train on enough
`ज्ञ`. So we deliberately manipulate exposure.

**The nine training runs.** Train the same architecture under three
conditions × three random seeds (Decision #14):

| Condition | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| Natural | ✓ | ✓ | ✓ |
| Flattened | ✓ | ✓ | ✓ |
| Inverted | ✓ | ✓ | ✓ |

Natural keeps the corpus histogram (common stays common). Flattened
pushes exposure toward equal. Inverted gives rare glyphs much more
exposure and common glyphs much less. **Total training volume stays
matched** — otherwise someone could say the flattened model was better
simply because it saw more data. The only major difference should be
*which* glyphs received the exposure. Three seeds are non-negotiable:
with one seed, the whole spread could be noise.

Then, for every grapheme cluster, we know how many times the model saw
it and how accurately it recognized it. Plot accuracy against (log)
exposure. If accuracy strongly increases with exposure, that is
evidence that **data exposure matters**.

But we also want: after accounting for exposure, are some glyphs still
systematically harder? Imagine `ज्ञ` is still substantially worse than
its exposure would predict. That leftover is the **residual** — a
candidate estimate of **intrinsic visual complexity**, not an
unexplained leftover. That residual is the point of the fit.

The orchestrator is `src/probes/probe1_exposure.py`. Real nine-run
results depend on Colab training against the manifests in
`data/manifests/`. Fake-data smoke proves orchestration only.

### Probe 2 — What does the model almost say?

Normally we look at the final answer: image → `क`. Internally the
model might have probabilities like `क` 52%, `ख` 35%, `ग` 8% — which
is a very different state of mind from `क` 99% and everything else
noise. In the first case it was **confused between `क` and `ख`**.

Probe 2 makes that concrete **against ground truth**. Align the
greedy prediction to the GT grapheme sequence; at every substitution,
record the true cluster, the predicted cluster, the top-5, and —
because we own the softmax — the probability and rank of the
**correct** cluster even when it was not selected. If that mass is
consistently real (say, rank 2–3 with non-trivial probability), the
encoder is carrying signal that argmax/confidence readout fails to
surface. If it is near floor, the hopeful story is false — report
that honestly.

A closed API often returns only `"क"`. Our instrument exposes the
full distribution (`generate.py`). That is one of the biggest reasons
we built it. Code: `probe2_confusion_graph.py` (DECISIONS.md #57);
analysis: `docs/probe2_confusion_analysis.md` (numbers after Colab).

### Probe 3 — Is it actually looking at the image?

Give the model three kinds of input of the same size:

- **A** — a real line image (`भारत`)
- **B** — a completely blank / white image
- **C** — random noise matched to the crop’s mean and variance

We would expect high confidence on A and low confidence on B and C,
because there is nothing to read. Suppose instead we get something
like real 99.29%, blank 98.98%, noise 98.77%. That is extremely
suspicious: the model is incredibly confident even when there is
nothing to read.

That suggests it may be relying on a **language prior** — “given what
I’ve seen during training, what text is likely?” — rather than “what
does this image actually contain?” The output can look perfectly
fluent. That is **guessing rather than reading**, and it is more
dangerous than an obvious failure (`??????` at 20% confidence),
because a user may trust a fluent, high-confidence wrong sentence.

Code: `probe3_blank_control.py`. **Verified** on disk: 9 Hindi files
(`natural`/`flattened`/`inverted` × seeds 0–2), n=100 each under
`data/probe_results/`. Natural seeds sit at ~0.99 confidence on real,
blank, and noise with a real−blank gap of only a few thousandths —
essentially no image grounding. Flattened/inverted drop mean
confidence but keep a **negative** real−blank gap (blank higher than
real). See `docs/results_analysis.md`.

### Probe 3b — Does more training teach it to look?

Probe 3 alone cannot separate “structurally ungrounded confidence”
from “simply undertrained.” Probe 3b re-runs the real-vs-blank
comparison at intermediate checkpoints (requires
`train.py --keep-snapshots`).

**Verified** on hindi/natural seeds **0, 1, 2**
(`data/probe_results/probe3_curve_hindi_natural_seed{0,1,2}.json`,
5 snapshots at steps 500/1000/2000/3000/5000; analysis in
`docs/probe3_curve_analysis.md`):

Across seeds, loss falls ~17× while mean real confidence **rises**
toward ~0.99 and accuracy stays near floor until late (mean acc
~0.13 only at step 5000). The real−blank gap **sign flips at 4 of 5
steps** and |gap SD| exceeds |gap mean| at 4 of 5 steps — it is
defensible to call the gap **indistinguishable from zero** across
training seeds (not an emerging vision signal). Step 3000 is
negative in all three seeds (mean −0.0075) as a separate
observation, without over-claiming a blank>real reversal.

The important framing: undertraining and ungrounded confidence are
**not** competing explanations. The model **is** undertrained. The
finding is that confidence gives **no indication** of that — a
calibrated undertrained model would report **low** confidence; this
one reports ~0.99 and rises as training proceeds. Code:
`probe3_training_curve.py`. See also
`docs/statistical_repair.md`.

### Probe 4 — Are we being fair when we call something wrong?

This reuses Chapter 1’s evaluation machinery: Tier 0 → Tier 1 →
Tier 2 → human residual. We do not want a different definition of
“wrong” for our own model than we used on Tesseract. If the instrument
produces an alternative valid Indic encoding, we must not incorrectly
punish it. Probe 4 = apply the same fairness rules to our instrument.
No new philosophy — just consistency.

### Probe 5 — Does confidence actually mean anything?

If the model says confidence = 70%, then across many examples about
70% of those predictions should actually be correct. That is
**calibration**.

A well-calibrated model’s 50% / 70% / 90% / 99% buckets have roughly
those accuracies. A badly calibrated model might sit at 99% confidence
while being right 10% of the time — wildly overconfident. That is a
huge practical problem.

**Why combine Probe 5 with Probe 1?** We already know some glyphs were
starved of exposure. Ask: does the model become particularly badly
calibrated on glyphs it rarely saw? High exposure might look like
confidence 90% / accuracy 88%; low exposure like confidence 95% /
accuracy 30%. That is a much more useful finding than “our model has
60% accuracy.” It tells us **when not to trust the model**. That
crossing is the centerpiece question of the project.

Code: `probe5_calibration.py`, with aggregation in
`src/analysis/aggregate_probe_results.py`. **Verified** on disk: 9
Hindi files, n=100 records each. Natural condition: nearly all mass
in the 0.9–1.0 confidence bucket while accuracy there is ~0.14–0.24
(ECE ≈ 0.75–0.85). Flattened/inverted lower mean confidence but
accuracy collapses to ~0–1%. See `docs/results_analysis.md`.

### Probe 5b — What if exposure is literally zero?

Even stronger: train on Hindi / Devanagari, then give the model
Santhali (Ol Chiki) or Kashmiri (Perso-Arabic) — scripts it has
**never** seen during training, not under-sampled, *absent*. If it
still says “I am 97% confident,” that is the sharpest version of
confidence ≠ knowledge.

**Verified** on hindi/natural seeds **0, 1, 2**
(`data/probe_results/probe5b_hindi_natural_seed{0,1,2}.jsonl`, 720
records total; analysis in `docs/probe5b_analysis.md`):

| Condition | seed0 | seed1 | seed2 | Mean | SD |
|-----------|-------|-------|-------|------|----|
| hindi | 0.9824 | 0.9897 | 0.9899 | 0.9873 | 0.0043 |
| santhali | 0.9848 | 0.9847 | 0.9876 | 0.9857 | 0.0016 |
| kashmiri | 0.9901 | 0.9894 | 0.9887 | 0.9894 | 0.0007 |
| blank | 0.9814 | 0.9933 | 0.9824 | 0.9857 | 0.0066 |

Mean confidence stays high everywhere. The **lead** equivalence
claim is threshold-free: between-condition range of across-seed
means is **0.0037**, smaller than hindi's across-seed SD (0.0043)
and blank's (0.0066). TOST at δ = 0.05 agrees, but does not need to
carry the claim alone.

A seed-0 Bonferroni pass for Kashmiri vs Hindi (z ≈ 2.54) is
**retracted** — it does not replicate (seed 1: hindi exceeds
kashmiri). Decision #14 earned its keep (DECISIONS.md #53).

Charset composition is the sharp signal: **360/360**
Santhali+Kashmiri images across all seeds emitted **zero**
characters of the correct script — the model writes fluent
Devanagari instead. Accuracy is deliberately not scored on unseen
scripts (DECISIONS.md #50). See `docs/statistical_repair.md`.

What this does **not** establish: that production OCR APIs behave
identically, or that more Hindi training would fix zero-shot
calibration. Code: `probe5b_zeroshot_floor.py`.

### Probe 6 — Did our synthetic world lie to us?

Chapter 2 created synthetic images for experimental control. The
danger: the model performs well on beautiful synthetic pages and fails
on real documents. Compare:

- **Tier A** — synthetic + clean
- **Tier B** — synthetic + blur / noise / skew / show-through
- **Tier C** — actual scanned documents

Imagine 92% / 75% / 48%. That says the controlled experiment may not
transfer well to reality. Probe 6 is a **reality check on the
experimental setup itself**.

**Paper scope (Decision #58), VERIFIED.** Compare Probe 3/5 synthetic
Claim B numbers to Tier C Hindi plain + degraded + blank. Eval images
are GlotOCR renders resized to 70 px, not photographs (Decision #74).
Held-out check: **0** overlapping ids between
`data/manifests/hindi_*.jsonl` and `data/raw/hindi/images/`. Pooled over
three natural seeds: mean confidence real_plain **0.9861**, real_degraded
**0.9768**, blank **0.9799**; Tier 1/2 line accuracy **0.0** on the
scored sample. The high-confidence / no-reading pattern is not an
artifact of the synthetic training crops alone.
Outputs: `data/probe_results/probe6_synthetic_real_hindi_seed{N}.jsonl`;
`docs/probe6_synthetic_real_analysis.md`. Unit tests 5/5.

### Attention ablation — does encoder memory move the peak?

Probe 3 is behavioral (blank vs text). Ablation is mechanistic: set
encoder output to zeros *before* `memory_projection` and decode again
(Decision #56). Mean confidence is a **free-run** under zero memory.
KL / top-1 / prior-sufficiency re-score the zero-memory decoder under
the **full-memory greedy token path** — same prefixes, so sequence
divergence does not confound the comparison. That is *not* teacher
forcing on ground truth (`docs/training_config.md` Step 0a).

**Verified** on the Probe 5b Hindi sample, three seeds:
mean confidence full **0.9861** vs zero-memory **0.9891** (Δ **−0.0030**);
top-1 agreement **0.8794**; prior sufficiency **0.8827**; mean
KL(full||zero) **1.075**. Interpretation: the confidence *peak* is
prior-dominated; only about 12% of token-choice mass depends on image
content. Agreeing steps keep a near-unit peak; the KL is concentrated
in the flips. Code: `src/probes/probe_attention_ablation.py`;
`docs/attention_ablation_analysis.md`.

### GT-likelihood — max-softmax is a biased estimator

Greedy confidence is *p*(argmax), which is high whenever the decoder is
peaky, even if the peak is the wrong token. Teacher-forced log *p*(GT)
and Shannon entropy of the full softmax are the bias-free check
(Decision #62). Same 60 Hindi + 60 blank × 3 seeds = **360** records.

Whole-sequence mean log *p*(GT): real **−1.783** vs blank **−1.751**.
First-token *p*(GT) ~10⁻¹⁰–10⁻¹² on **both**. Rest-of-sequence mean log
*p* ≈ **−1.14** either way (~0.32 probability). Entropy stays tiny
(real **0.0210**, blank **0.0253**). The dissociation — high
self-confidence, floor probability on the true first grapheme —
replicates under this estimator. Offline pairing with max-softmax and
the uniform 1/367 baseline is in
`docs/paper_defensibility_stats.md` (cite that file for tables).
Code: `src/probes/probe_gt_likelihood.py`;
`docs/gt_likelihood_analysis.md`.

### Noise and patch-scrambled input — bypass versus domain shift

Blank versus text-bearing can be dismissed as “the eval renderer is
just too far from training.” Gaussian noise and patch-scrambled real
images test whether the model responds to visual variation of any
kind. Pooled whole-sequence mean log *p*(GT): noise **−1.756**,
scrambled **−1.730**, against text-bearing **−1.783** and blank
**−1.751** — all four within 0.05 nats. Position 0 still collapses
(noise **−23.7**, scrambled **−24.1**). Pure noise has no structure a
domain-shift account would expect the model to process differently
from a familiar-but-shifted rendering. That is **evidence for bypass
over domain shift, not proof** that a visual representation never
existed: an instrument that never learned to use images under any
condition is consistent with both a total bypass and a domain-shift
failure with no residual visual sensitivity. Data:
`probe_gt_likelihood_extra_hindi_natural_seed{N}.jsonl`;
`docs/tier0d_noise_scrambled.md`. Figure files (not yet in the PDF):
`paper/figures/fig6_noise_scrambled.pdf`.

### Probe 5 AUROC is in-training-manifest only (split resolved, empty)

Section 9 / Probe 5’s AUROC **0.838** is **not** an open question
about held-out text. All 300 evaluation instances (100 strings × 3
seeds) are drawn from `hindi_natural.jsonl`. The non-matching subset
is n=0, so the memorisation-versus-correctness split **cannot be run**
on this evaluation set. The figure is an in-training-manifest result,
not evidence of generalisation. Details:
`docs/memorisation_split.md`. Future work is a new eval set withheld
before training, not a rerun of this probe.

### Offline defensibility battery and figures

No new GPU passes: `src/analysis/paper_defensibility_stats.py` writes
`docs/paper_defensibility_stats.md` (AUROC/Spearman, image×seed
variance, flattened/inverted accuracy, Stage 0 per-engine Tier 1,
position-0 vs uniform, Wilcoxon with clustering, seed-clustered
bootstrap 2,000 reps, length-controlled CER, n-gram sweep, equal-mass
ECE). `src/analysis/make_paper_figures.py` writes Figures 1–4 PDF under
`paper/figures/` and PNG under `docs/figures/` (Decision #64).

**Now measured (in the preprint):** noise + patch-scrambled
teacher-forcing (`docs/tier0d_noise_scrambled.md`); n-gram KL / argmax
agreement (`docs/ngram_kl_argmax.md`); Probe 5 memorisation split
empty (`docs/memorisation_split.md`); PaddleOCR same-protocol control
**not viable** (`docs/paddleocr_feasibility.md`).

**Authored, not run** (need checkpoints on disk): shuffled image–text
teacher forcing (`probe_gt_mismatch.py`), cross-attention contribution
norms (`probe_cross_attn_norms.py`). Log:
`docs/remaining_measurements.md`.

### How all the probes fit together

The project is not running six random experiments. They answer
different pieces of one question:

```text
                 INSTRUMENT
                     │
       ┌─────────────┼──────────────┐
       ↓             ↓              ↓
   EXPOSURE       CONFUSION      GROUNDING
   Probe 1        Probe 2        Probe 3
       │             │              │
 Does seeing      What does      Is it actually
 more help?      it confuse?     looking?
       │
       └──────────────┬──────────────┘
                      ↓
                 Probe 4
             Is our scoring fair?
                      ↓
                 Probe 5
          Can I trust confidence?
                      ↓
                 Probe 6
           Does this work in reality?
```

A possible scientific story (if the measurements hold) looks like:

> Rare glyphs receive less exposure → their accuracy drops → the
> model remains highly confident → blank-image tests show similar
> confidence → some failures are likely language-prior-driven rather
> than visual recognition → confidence can potentially identify cases
> that need escalation.

That is a much stronger conclusion than “our OCR model got 72%
accuracy.” One probe can be dismissed. Several probes that agree
become a diagnosis with an implied fix: change the training mixture,
change the decoding policy, or refuse to trust high confidence on
low-exposure scripts.

### The entire project in one chain

**Stage 0 — Define “wrong.”** Indic Unicode makes naïve OCR metrics
misleading.  
**Stage 1 — Control what the model sees.** You cannot study exposure
if exposure is uncontrolled.  
**Stage 2a — Build a model from scratch.** A pretrained model has
already seen Indic data, so you cannot isolate exposure.  
**Probes — Open the model up.** The six questions above.  
**Eventually — Apply the findings.** If confidence is useful, use it
as a router (Chapter 9): low confidence → stronger OCR / human; high
confidence → accept. But only after demonstrating that confidence
actually predicts errors.

If someone asks what this project is actually about, one sentence:

> We’re building a controlled OCR experiment to separate data-exposure
> effects from intrinsic script difficulty, and then probing a model
> we own to understand whether its predictions, confidence, and errors
> are actually grounded in the visual input.

That is the core of the entire repository.

> **What to remember.** The point of owning the model is not to brag
> about accuracy; it is to ask whether the model knows when it does not
> know.

---

## Chapter 8 — Why You Can’t Learn Everything From an API

Chapters 8 and 9 are the **“so what do we do with everything we
learned?”** part of the project.

> Chapters 0–7 = understand the problem and diagnose the model.  
> Chapter 8 = check whether the diagnosis applies to a real production
> OCR system.  
> Chapter 9 = turn the diagnosis into an operational decision: when
> should we trust the model, and when should we escalate?

### Why not just use a production OCR API?

Suppose you send an image to an OCR API. You might get back text, and
maybe a confidence score. That is useful for **evaluating the output**.
But you generally do not get access to the model’s internals.

You typically cannot ask:

- “For this grapheme, what were your top 10 possible predictions?”
  (Probe 2)
- “Now take the exact same image, replace it with a blank, and tell me
  what your model believes.” (Probe 3)
- “How many times did this model see this particular glyph during
  training?” (Probe 1)

You also cannot re-run a thousand threshold sweeps without paying per
page.

So there is a fundamental distinction:

**API**

```text
Image → [BLACK BOX] → Text
```

You can observe the input and output.

**Our instrument**

```text
Image → encoder → decoder → probabilities → text
```

We control the whole thing, so we can inspect what happens inside.

That is not a moral complaint about vendors. It is a measurement fact:
some questions are foreclosed entirely if you only ever call the API.

### So why use the API at all?

The instrument is useful for discovering things like “rare-glyph
exposure seems to cause this particular failure.” A legitimate
objection follows:

> Okay, you discovered that on your tiny 30–60M parameter model. Why
> should I believe this matters for a real production OCR system?

That is what **transfer** is asking. The instrument’s findings are
causal claims on a small model. Stage 5 asks whether those claims
**rhyme** with a production system’s error structure.

### What “transfer” means here

Suppose the instrument finds Glyph A at 10% error, B at 30%, C at 50%.
Evaluate a production system such as Sarvam on the **same kinds of
glyphs** and get 5%, 18%, 35%. The absolute numbers are not the same.
That is fine. The interesting question is:

> Do the same glyphs tend to be difficult for both systems?

```text
Instrument difficult glyphs
             ↕
       correlation?
             ↕
Production difficult glyphs
```

If yes, something learned from the small controlled instrument **might
reflect a real property of the production OCR problem**. If no, that
is also useful: “our causal findings on the small instrument don’t
transfer.” That is not a failed experiment. That is a scientific
finding.

### Why rank correlation?

We are not expecting instrument error = Sarvam error. The models are
completely different sizes and architectures. We care about **ranking**:
if both systems agree that A is easiest, B medium, C hardest, their
rankings correlate even when the percentages differ.

Stage 5 plans a **pre-specified rank-correlation test with a
permutation null** — choose the statistic *before* looking at results.
Otherwise you could look at the data first and then pick whatever
statistic makes the result look strongest.

### Why clean *and* degraded pages?

Suppose you only test on perfect synthetic pages. Almost every OCR
system may perform very well (97% vs 98%). There is not much
information there. Introduce blur, noise, skew, show-through, real
document variation, and systems start separating. Clean-only
comparison is close to meaningless: systems often cluster on clean
synthetic text and spread on real degradation (Decision #15).

### Why the ~200-page budget matters

This is not just an engineering detail. Suppose you have about 200
paid API calls. You could accidentally do 200 images × 10 confidence
thresholds × 5 experiments = 10,000 calls. The experiment becomes
impossible.

So Decision #19 is strict:

> Call the API **once per page**. Save the result. Never call it again
> for later analysis.

```text
                    Sarvam API
                       ↓
                    ~200 pages
                       ↓
                  CACHE (data/cache/)
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
     transfer       cascade        later
     analysis       sweeps         probes
```

Everything after the API call happens **offline**. Separate expensive
data collection from cheap analysis. An escalation sweep that re-called
the API per threshold would burn the whole budget on one experiment.

### What actually ran (Stage 5a), plus Stage 5b’s pre-registered null

**Stage 5a — built and run.** `src/eval/sarvam_client.py` wraps Doc-AI
**Extract** (not Digitise — only Extract exposes
`annotations.{field}.confidence`, confirmed docs.sarvam.ai, Decision
#59). Schema is a single field `{"full_text": "..."}` so we get one
page-level number comparable to the instrument’s `mean_confidence`.
Every page is cached under `data/cache/sarvam/` by SHA-256 of image
bytes; re-runs never re-call (Decision #19).
`src/probes/sarvam_transfer_probe.py` drew 35 images with `Random(0)`:
10 Hindi / 10 Santhali / 10 Kashmiri / 5 blank. Budget used: **₹17.50**.
Output: `data/probe_results/sarvam_transfer_probe.jsonl`. Analysis:
`docs/sarvam_vision_confidence.md` (Decision #77: one Sarvam-facing
note, not a second API doc).

Published accuracies (re-verified sarvam.ai/blogs/sarvam-vision,
Decisions #6 / #60): Hindi **95.91%**, Santhali **80.32%**, Kashmiri
**55.93%**. Headline: Hindi → Kashmiri accuracy gap **39.98 pp**;
Extract confidence **0.9997 → 0.9970** (Δ **0.0027**). Blanks score
**0.0000**. Page-level confidence does **not** track the published
language gap, even though the API will go to zero on empty pages.

This Stage 5a result is **not** in the preprint. It is also **not** the
glyph-class transfer of Decision #15. The unit that *did* run as Stage
5b is the **per-image** Spearman test locked in Decision **#89**.

**Stage 5b — pre-registered, computed, null.** Statistic locked before
looking at ρ: Spearman between instrument difficulty (−mean over seeds
of teacher-forced `mean_log_p_gt` on `condition=real`) and production
Tier-1 grapheme CER, with a 10,000-shuffle permutation null (seed 0).
Code: `src/eval/transfer_analysis.py --compute-now`. Report:
`docs/stage5b_rank_correlation.md`. Evidence JSONL includes
`data/probe_results/sarvam_stage5b_pages.jsonl` plus Stage 5a cache.

Primary (Hindi plains, Sarvam CER, **n=60**):

- Spearman ρ = **0.0293**
- permutation p = **0.8267**

That is **essentially zero** correlation between the instrument’s own
difficulty ranking and Sarvam’s real production error on the same
images. Treat this as a **genuine null**, not an inconclusive or failed
experiment: Decision #89 locked the protocol specifically so a null
like this one would be trustworthy rather than explained away.

Secondaries (same protocol; full numbers in the report):

- Tesseract plains: ρ **−0.0808**, p **0.5453**, n=60 — null / wrong-signed
- PaddleOCR plains: ρ **−0.1593**, p **0.2173**, n=60 — null / wrong-signed
- Hindi degraded, Sarvam: ρ **−0.2222**, p **0.2913**, **n=24** — null /
  wrong-signed (see budget note: this n is underpowered, not a
  confirmed null at the pre-registered n=60)
- Surya plains: ρ **0.2348**, p **0.0685**, n=60 — the **closest thing
  to a signal, not a finding**. Positive direction; does **not** cross
  the pre-registered significance threshold. Do not round p=0.0685 down
  to “significant.”

Exploratory and **explicitly underpowered** (state that in the text,
not only in a footnote): Santhali and Kashmiri use −Probe 5b
`mean_confidence` as the instrument axis because teacher-forced log
p(GT) does not exist for those scripts on this instrument, and each arm
is only **n=10**. Santhali ρ **−0.3891**, p **0.2632**; Kashmiri ρ
**−0.5394**, p **0.1182**. n=10 cannot support a claim either direction.

**Budget exhaustion (honest, non-speculative).** Of 110 new Stage 5b
Extract attempts (`--set both`), **36 failed with HTTP 402**
(insufficient credit). All 36 were in the **Hindi degraded** condition —
plain pages completed in full before the balance ran out. That leaves
the Hindi-degraded secondary at **n=24** instead of the pre-registered
n=60. Santhali/Kashmiri exploratory arms were never budgeted past n=10
inside the project’s ₹100-ish cap. What full budget would have changed
is **statistical power**, not a known result: the degraded secondary
would have reached its pre-registered n, and the exploratory arms could
in principle have been re-scoped as primary tests with adequate n. We
do **not** know what those additional points would have shown — only
that current n=24 and n=10 results are underpowered rather than
“confirmed null,” unlike the fully powered **n=60 primary**. The 402
failures were correctly excluded from every computed statistic
(explicit `error` field, null confidence — not a fake zero); that is
why reported n values differ from 110 rather than silently including
failed rows.

**Coherence with the paper’s central finding.** Stage 5a (confidence
does not track published language accuracy) and Stage 5b (instrument
difficulty does not predict Sarvam CER) independently corroborate, on
**real production data**, the same story the preprint tells about the
owned instrument: a confidence / difficulty signal that is not grounded
in visual reading also does not transfer to predicting real-world
hardness. That is a useful negative result, not a wasted probe. Stage
6 (Chapter 9) makes the same point for triage.

If you explain Chapter 8 in a conversation, do not reduce it to “we’re
comparing our model against Sarvam.” Say: the owned model is how we
ask *why*; Stage 5a asks whether a production confidence field tracks
that system’s own published accuracy spread; Stage 5b asks whether the
instrument’s per-image difficulty ranking predicts Sarvam’s CER on the
same Hindi plains — and the pre-registered answer is no (ρ≈0.03,
p≈0.83).

> **What to remember.** An API can score a language; only a model you
> own can tell you *why*. On production Extract data, confidence need
> not move when published accuracy does — and the instrument’s own
> difficulty ranking does not predict Sarvam’s per-image error either
> (Stage 5b primary null, Decision #89).

---

## Chapter 9 — Deciding When to Trust a Machine and When to Ask for Help

Now suppose Probe 5 tells us something interesting: model confidence
is useful for predicting whether an OCR result is correct. We can
potentially turn that into a **routing system**.

### What is selective prediction?

Imagine the small model processes a page. Easy page, confidence 98% —
maybe we trust it. Difficult page, confidence 35% — instead of forcing
an answer, we say: “I’m not confident enough. Send this somewhere
better.”

That is **abstention** or **selective prediction**. In this project’s
terminology, **abstention = escalation**: send the page to a stronger
(and more expensive) OCR stack.

```text
                    Image
                      ↓
                Small OCR model
                      ↓
               Confidence score
                 /           \
          High confidence   Low confidence
                ↓               ↓
             Accept       Strong OCR / human
```

The small model becomes a **router**. It does not need to be the best
OCR system. It needs to be good at answering: “Do I think this
particular page is safe for me to handle?” That is a very different
objective from headline accuracy.

Escalation only helps if the signal you escalate on is meaningful. A
random coin flip is an escalation policy; it is just a bad one. Probe
5b’s confidence scores are the candidate signal for Stage 6.

### Why not just escalate everything?

Then the small model is useless. Every page → expensive OCR. Accuracy
might be excellent, but you have gained nothing from the router. We
want something like: 100 pages through the small model, 80 handled
confidently, 20 sent to expensive OCR, while recovering most of the
accuracy of the expensive system.

### How we tested the router (and what it showed)

Using the Stage 5 cache only (no new paid calls), Stage 6 holds the
escalation **count** fixed and asks which policy leaves the lowest
residual system CER when escalated pages are scored as perfect
(CER=0). Code: `src/probes/cascade.py --compute-now`. Report:
`docs/stage6_triage_cascade.md`. Cohort: **n=60** Hindi plains with
Sarvam CER.

Policies compared at every k = 0…n, with the fair headline slice at
**k = round(0.2 × n) = 12** (20% of pages):

1. **Instrument** — escalate the k *lowest* Probe 5b `mean_confidence`
   (3-seed mean, Hindi).
2. **Random** — mean residual over 1000 draws of k pages (RNG seed 0).
3. **Layout** — escalate the k *longest* GT grapheme strings.
4. **Tesseract** — escalate the k lowest Tesseract confidences on the
   matching `*_plain.png`.

At that pre-specified fair point:

| Policy | Residual system CER |
|---|---:|
| Instrument confidence | **0.0109** |
| Random (mean of 1000) | **0.0095** |
| Layout (GT grapheme count) | **0.0092** |
| Tesseract confidence | **0.0079** |

Lower is better. On this test, **instrument-confidence routing is worse
than random**, worse than routing by GT grapheme length, and clearly
worse than Tesseract’s own confidence. That pattern holds across most
of the full k-sweep in the report, not only at k=12. State plainly:
the instrument’s confidence signal is **not** a useful routing signal
for real production triage here.

### Why aren’t we claiming “we’ll save money”?

This is a subtle but important distinction (Decision #16).

The instrument itself is **not expected to be better OCR than a
production system**. Saying “our model will reduce OCR costs by 50%”
would be a huge claim that is not justified. A cost-savings story
built on a weak base model dies under one follow-up question.

The defensible claim is **router quality**:

> Our model’s confidence is a useful signal for identifying pages that
> should be escalated.

Stage 6’s answer on this cohort is **no** — which is still a scientific
result. A router-quality story survives even when the signal fails,
because you measured the **confidence signal**, not a business slide.

### How Chapters 8 and 9 connect — and the coherence note

```text
                 OUR INSTRUMENT
                       │
                       ↓
                     PROBES
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
   Exposure        Confidence       Errors
                       │
                       ↓
                Transfer to
              production OCR
                       │
               "Does it rhyme?"
                       │
                 If confidence
                  is useful
                       │
                     ROUTER
                    /        \
               Accept      Escalate
                              │
                        Strong OCR /
                           human
```

Stage 5b’s primary null (instrument difficulty ≉ Sarvam CER) and Stage
6’s negative routing result (instrument confidence worse than random at
matched k) **independently corroborate** the paper’s central finding on
**real production data**, not only on the synthetic instrument: a
confidence signal that does not track genuine visual grounding also
does not transfer to predicting real-world difficulty or improving
real-world triage. That is a real, useful negative result, not a wasted
probe.

The project’s evolution, end to end:

**Stage 0** — How do we define an OCR error?  
**Stage 1** — How do we control what the model sees?  
**Stage 2** — How do we build a model whose internals we control?  
**Probes** — What is the model actually learning?  
**Stage 5 / Chapter 8** — Do those findings tell us anything about
production OCR? (5a: confidence vs published language gap; 5b: null
rank transfer.)  
**Stage 6 / Chapter 9** — Can we use the model’s confidence to decide
when to trust it? (On this test: no — worse than random at k=12.)

If you explain Chapter 9 in a conversation:

> Stage 6 asks whether Probe 5b confidence identifies hard Hindi plains
> better than random, layout length, or Tesseract confidence at the
> same escalation count. At k=12 / 20%, residual CER is 0.0109 for the
> instrument vs 0.0095 random and 0.0079 Tesseract — so the signal is
> not useful for production triage here, which coheres with Stage 5b’s
> null transfer and with the paper’s “confidence without grounding”
> story.

> **What to remember.** Confidence is only useful if it knows when to
> hand the page to someone else — and on this Stage 6 test, the
> instrument’s confidence does not. That negative is the finding.

---

## Conclusion

Step back from the modules and look at what the project is arguing —
and what it actually measured.

**First**, published Indic OCR numbers are hard to interpret until you
separate encoding variants from genuine misreads. Stage 0 shows that on
real Tesseract output, on the order of one in five non-exact predictions
is a Tier 1 variant, not a wrong reading — and a large unreviewed slice
means that fraction stays **provisional** until adjudication (Decision
#55).

**Second**, “this language is hard” is not one claim. It mixes how often
glyphs were seen with how hard they are to see. Stage 1 turned exposure
into a dial (TV ≤ 0.08). Flattened/inverted training then **failed to
read**, so the causal slope β is withheld (Decision #49). The dial
worked as apparatus; it did not yield the originally hoped-for
within-glyph exposure experiment on this compute budget.

**Third**, the only way to ask whether a model is reading or guessing —
and whether its confidence means anything — is to own a model you can
open. The instrument does **not** read its held-out GlotOCR lines (CER
near 1 on text and on blank). Confidence stays near ceiling anyway:
blank, unseen scripts, encoder ablation, teacher-forced position 0.
That is the preprint’s result: a position-resolved profile of
confidence without correctness. Implied fix: do not route human review
on max-softmax for this decoder family without a grounding check.

**Fourth**, Stage 5a shows a production Extract confidence field that
barely moves across a ~40 pp published accuracy gap, while scoring
blank pages at 0. That rhymes with Claim B without claiming we opened
Sarvam’s encoder. Stage 5b’s pre-registered per-image Spearman test
is a genuine null (ρ≈0.03, p≈0.83 on n=60 Hindi plains), and Stage 6
instrument-confidence routing is worse than random at the fair 20%
escalation point — both on real production data, both outside the
preprint. Glyph-class transfer (#15), the demo model, and RLVR remain
how the owned stack would meet a stronger reader; those are separate
from the 5b/6 negatives already measured.

If you only remember three implied fixes from the whole book, remember
these:

1. Score Indic OCR with grapheme-aware, Tier-1-aware metrics.
2. Treat glyph exposure as something you design, not something you
   observe after the fact — and withhold causal slopes when the
   reader never trained.
3. Treat fluent, high-confidence failure as a calibration and routing
   problem — not only as “need more accuracy.” Position 0 is the test.

That is the scientific spine of the repository.

---

## What I would improve about this methodology

This is not a new stage. It is the honest answer to “what is weak in
your own design?” Decision #46 recorded the list. Several items later
landed as **offline** analyses in `paper_defensibility_stats.py`
(Decision #64). Below, **done** vs **still open** is explicit so this
section does not pretend 2026-08 plans are still the present.

### 1. Fixed-width confidence bins are broken at this sample size

`probe5_calibration.py` buckets with `np.linspace(0, 1, 11)` — ten
fixed-width bins — and prints every occupied bin the same way:

```text
conf {range}: acc=… (n=…)
```

Default `--n-samples` is 30. At that N, some bins will be empty and
some will contain one prediction. A singleton bucket (`n=1`, accuracy
0.00 or 1.00) looks as authoritative in the table as a bucket with
`n=18`. It is not.

The standard fix is **adaptive (equal-frequency) binning**: sort
predictions by confidence and split into equal-sized groups so every
bucket carries the same statistical weight. Pair that with one summary
number, **Expected Calibration Error (ECE)**, and ideally the **Brier
score** too — ECE is known to be sensitive to how you choose bins;
Brier score is not.

**Partial, done offline:** equal-mass ECE (10 bins) is in
`docs/paper_defensibility_stats.md`. The live `probe5_calibration.py`
printer is still fixed-width. Brier score is still open.

### 2. Make Probe 3 mechanistic, not only behavioral

Right now Probe 3 says “confidence barely changes on blank / noise” —
a conclusion about the *output*. The instrument already has a ViT
encoder and decoder cross-attention. Nothing stops you from extracting
cross-attention weights at each generation step: does attention
actually move toward the glyph being generated, or stay diffuse even
on real images? If it is diffuse on real text, that is a second,
independent line of evidence for the same “not reading” conclusion, at
the level of mechanism rather than outcome.

This requires **zero retraining** — introspection on an existing
forward pass. Related techniques in vision: occlusion sensitivity
(Zeiler & Fergus) and attention rollout for transformers. Probe 3
today implements only `make_blank` and `make_matched_noise`.

**Partial progress (DECISIONS.md #56), now VERIFIED:**
`src/probes/probe_attention_ablation.py` ablates encoder *content*
(zeros before `memory_projection`) rather than reading attention
weights. That answers a narrower but Claim-B-central question — does
confidence change when the decoder gets no image features at all? —
and is complementary to weight-map introspection, which remains open.
See `docs/attention_ablation_analysis.md`. Cross-attn *contribution
norms* (`probe_cross_attn_norms.py`) are authored but not run
(`docs/remaining_measurements.md`).

### 3. Add a stricter control than blank / matched noise

Blank and matched-noise are good baselines. A sharper one is
**patch-shuffle**: take a real line image, keep every 14×14 patch
pixel-identical, permute their spatial arrangement, then feed it to
the encoder. That destroys global structure while preserving local
texture statistics exactly — stricter than noise, which has no texture
at all. If likelihood survives patch-shuffling too, the finding is
more specific: not just “isn’t reading,” but “isn’t even using local
stroke shape.”

**Now run.** `probe_gt_likelihood.py --extra-conditions noise scrambled`
on three seeds (`docs/tier0d_noise_scrambled.md`; numbers in
`paper/main.tex` and the Chapter 7 subsection above). Patch-shuffle
here is the extra-condition scramble, not a separate Probe 3
confidence tile.

### 4. Stop reporting one seed where Decision #14 already required three

**Mostly done.** Probe 3b, 5b, ablation, GT-likelihood, Probe 6, and
the defensibility battery report three-seed mean±SD. Decision #53
exists because a seed-0 Kashmiri Bonferroni pass did not replicate.
Do not regress to seed-0-only headlines in new write-ups.

### 5. Probe 1’s regression should respect nesting

The planned fit is a plain fixed-effects regression (accuracy ~
log(exposure) + glyph class). That treats every observation as
independent. It is not: observations are nested within glyph class,
within seed, within condition. A **mixed-effects model** (random
intercepts per glyph class and per seed) accounts for that structure
and would give real intervals on the complexity residual rather than a
bare point estimate per glyph.

### 6. One human labeling the residuals is a nameable gap

Stage 0’s remaining genuine-error labels go through one person, with
`hand_review_assist.py` suggesting and the human confirming. The
standard fix in annotation-heavy eval is a second independent labeler
on a random subsample (even 15–20 cases) and **Cohen’s kappa**. That
does not require relabeling the whole UNREVIEWED pile. It shows the
taxonomy is reproducible, not one person’s judgment calls.

### 7. Effect sizes with bootstrap intervals, not significance stars

With three seeds, a p-value is mostly decoration. The more defensible
framing at this N: bootstrap over the ~30 probe samples within a run,
report effect sizes with confidence intervals, and say explicitly that
you are not claiming statistical significance — you are reporting a
directional effect size with honestly stated uncertainty.

**#7 effect sizes:** seed-clustered bootstrap CIs (2,000 reps) and
clustered Wilcoxon tests are in `docs/paper_defensibility_stats.md`.
Mixed-effects Probe 1 (#5) is still open and less urgent while β is
withheld. Kappa on hand-review (#6) is still open.

> **What to remember.** Several #46 upgrades landed as offline stats.
> Still weak: live Probe 5 binning, attention-weight maps, patch-shuffle
> *runs*, second annotator, local checkpoint availability for new
> forward passes.

---

## Appendix A — Decisions index

Full write-ups: `DECISIONS.md`. Summaries: **Decision catalog** near
the front of this file.

| ID | Topic | Where in this book |
|---|---|---|
| 1 | Instrument vs demo | 0, 3, 4; Q2–Q3 |
| 2 | Grapheme-cluster vocabulary | 1, 3 |
| 3 | Demo base = SmolDocling (#84) | 4; Stage 2b status |
| 79–84 | T4 not faked; SFT corpus; bank tau; RLVR; mistral3 ids; #3 close | 4–6 |
| 4 | NFC already upstream | 1 |
| 6 | Script scope + Sarvam verify | 0, 7, 8; Q7 |
| 7 | Grapheme-level alignment | 1 |
| 8, 18, 54 | Tier 2 ISO 15919; scope; validation n | 1 |
| 9–10, 24–30 | Renderer layouts, degradation, frequency | 2 |
| 11 | RLVR coverage ablation only | 6 |
| 12–13 | Table binding; Extract | 5, 8 |
| 14, 53 | Three seeds; Kashmiri retraction | 7; Q3 |
| 15–16, 19, 59–60, 77 | Transfer, cascade, cache, Extract, blog cite | 8, 9 |
| 20–23, 26, 35, 55 | Hand-review + adjudication | 1 |
| 31–34, 42 | Baselines resume, timeouts, Paddle | 1; How built |
| 36–41, 43–45, 47–48, 65–66, 73 | Instrument, crops, snapshots, no holdout | 2, 3, 7 |
| 46, 64 | Methodology list; offline battery | after Conclusion |
| 49 | Withhold Probe 1 β | 7; Q2 |
| 50–52 | 5b no unseen CER; Bonferroni; TOST | 7 |
| 56 | Attention ablation protocol | 7 |
| 57 | Probe 2 GT-aligned + p(true) | 7; Q4 |
| 58, 74 | Probe 6 paper scope; never say “real” | 7; Q6 |
| 61 | jsonl in git; zips local | How built |
| 62–63 | GT-likelihood; follow-up probes | 7 |
| 67–72, 75–76 | Paper dir, Pages, figures, README | paper/, README |

---

## Appendix B — Built vs described-only

**Built and used for measured claims:** Stage 0 taxonomy stack
(Tesseract + Surya + PaddleOCR n=420); Stage 1 renderer + Hindi/Bengali
manifests; Stage 2a instrument (9 Hindi checkpoints on Colab);
`make smoke-test`; Probe 1 FE (β withheld); Probes 2, 3, 3b, 5, 5b;
attention ablation; GT-likelihood including noise/scrambled extras;
n-gram KL; Probe 5 overlap check; Probe 6 paper scope; Stage 5a Extract
35-page probe; **Stage 5b** rank-correlation transfer (offline;
`docs/stage5b_rank_correlation.md`); **Stage 6** triage cascade
(offline; `docs/stage6_triage_cascade.md`); offline
`paper_defensibility_stats.md` + paper figures.

**Preprint scope vs book:** The PDF covers the instrument and
mechanistic probes. Stage 5b and Stage 6 are subsequent extensions —
documented here and on the Extract tab of `index.html`, not in
`paper/main.tex`.

**Partial:** layout bank (`form` / `table-embedded`, india.gov);
degradation source mix; UNREVIEWED / unlabeled
adjudication sample; Probe 4 on instrument outputs; Bengali probe sweep;
equal-mass ECE offline but not in the live Probe 5 printer;
`analyze_sarvam_transfer.py` 5a bootstrap CIs.

**Code written, not run (need checkpoints):** `probe_gt_mismatch.py`,
`probe_cross_attn_norms.py`, position-0 null rank
(`docs/remaining_measurements.md`). Surya and PaddleOCR
instrument-matched controls are **not viable** (architecture), not
waiting on a run.

**Described only (this phase):** RLVR **policy** training (reward +
SFT exist; `docs/rlvr_scoping.md`); reading-order / table metrics on
the **demo** (code exists, no demo curve); mixed-effects Probe 1;
second annotator kappa. Demo LoRA **SFT** ran on Colab T4.

---

## Appendix C — Language / idiom guide

| Construct | Why it appears |
|---|---|
| `regex` `\X` | Grapheme clusters; stdlib `re` cannot do this |
| NFC then NFD | Compose to compare; decompose to strip matras |
| Largest-remainder counts | Exact glyph bags for the frequency dial |
| Bigram packing | Keep some language structure under a hard quota |
| Ink projections | Layout from scans without a second neural net |
| Laplacian → blur σ | Turn a sharpness measurement into a PIL apply unit |
| Append+skip JSONL | Resume after Colab death without corrupting results |
| fp16, not bf16 | Free Colab T4 is Turing |

---

## Appendix D — Glossary

| Term | Plain meaning |
|---|---|
| Grapheme cluster / akshara | One visual syllable; may be many code points |
| Matra / nukta / virama | Vowel sign / dotted consonant mark / join killer |
| Tier 1 / Tier 2 | Encoding equivalence / phonetic equivalence |
| TV distance | How far two glyph histograms are from each other |
| Instrument | From-scratch model built to be probed |
| Demo | Pretrained+LoRA model built to look like production |
| Tier A / B / C | Clean controlled / degraded / real pages |
| Calibration | Whether confidence matches actual correctness |
| Kendall tau | How disordered a reading order is |
| RLVR | Reinforcement learning with automatically checkable rewards |

---

## Appendix E — Reproduce every headline number

From the repo root. Prefer these over memory.

```bash
# Stage 0 fractions (measured headline ~20.4% Tier 1 among Tesseract non-exact)
python3 src/eval/error_taxonomy.py

# Tier / assist self-tests
python3 src/eval/equivalence_tables.py
python3 src/eval/hand_review_assist.py
PYTHONPATH=src/eval python3 src/eval/hand_review.py --self-test

# Glyph-frequency TV gate on Hindi GT
PYTHONPATH=src python3 -c "
import json, numpy as np
from renderer.glyph_frequency import resample_corpus
texts=[json.loads(l)['text'] for l in open('data/raw/hindi/ground_truth.jsonl')]
for m in ('natural','flattened','inverted'):
    r=resample_corpus(texts,m,rng=np.random.default_rng(0))
    print(m, r.tv_distance, r.within_tolerance())
"

# Architecture smoke (no scientific findings)
make smoke-test

# Unit tests
pytest -q

# Paper defensibility statistics (offline battery, 9 analyses + reviewer items)
PYTHONPATH=src/eval:src python3 src/analysis/paper_defensibility_stats.py \
  --results-root data/probe_results \
  --manifest-root data/manifests \
  --out docs/paper_defensibility_stats.md \
  --plot docs/figures/gt_likelihood_position_curve.png

# Paper figure generation (dual PDF for paper/figures and PNG for docs/figures)
PYTHONPATH=src/eval python3 src/analysis/make_paper_figures.py \
  --results-root data/probe_results \
  --out-dir docs/figures \
  --paper-dir paper/figures \
  --seeds 0 1 2
```

Probe 3/5 aggregates need a checkpoint directory (often on Colab Drive):

```bash
python3 src/probes/probe3_blank_control.py --manifest data/manifests/hindi_natural.jsonl \
  --output-root <checkpoints> --condition natural --seed 0 --n-samples 30 \
  --out data/probe_results/probe3_hindi_natural_seed0.jsonl
python3 src/probes/probe5_calibration.py --manifest data/manifests/hindi_natural.jsonl \
  --output-root <checkpoints> --condition natural --seed 0 --language hindi --n-samples 30 \
  --out data/probe_results/probe5_hindi_natural_seed0.jsonl
python3 src/analysis/aggregate_probe_results.py --script hindi --out data/probe_results/summary_hindi.json
```

---

## Appendix F — Where every number lives

Do not memorize headlines. Use this lookup (same as `docs/RESULTS.md`).

| Claim | Canonical write-up |
|---|---|
| Stage 0 Tier 1 among Tesseract non-exact | `error_taxonomy.py` printout; `docs/adjudication_analysis.md` |
| Glyph-frequency TV | `glyph_frequency.py`; IMPLEMENTATION Stage 1 |
| Probe 1 β withheld | `docs/probe1_fixed_effects.md` |
| Probe 2 confusion | `docs/probe2_confusion_analysis.md` |
| Probe 3 / 5 aggregates | `docs/results_analysis.md` |
| Probe 3b curve | `docs/probe3_curve_analysis.md` |
| Probe 5b + retractions | `docs/probe5b_analysis.md`, `docs/statistical_repair.md` |
| Ablation | `docs/attention_ablation_analysis.md` |
| GT-likelihood | `docs/gt_likelihood_analysis.md` |
| Paper tables / CER / pos-0 | `docs/paper_defensibility_stats.md` |
| Training hyperparameters | `docs/training_config.md` |
| Probe 6 | `docs/probe6_synthetic_real_analysis.md` |
| Stage 5a Sarvam | `docs/sarvam_vision_confidence.md` |
| Stage 5b Spearman transfer | `docs/stage5b_rank_correlation.md` |
| Stage 6 triage cascade | `docs/stage6_triage_cascade.md` |
| Remaining GPU work | `docs/remaining_measurements.md` |
| Colab log | `COLAB_RUNS.md` |

Raw jsonl: `data/probe_results/`. Preprint: `paper/main.pdf`. Project
page: `index.html` (Decision #69; default view is the preprint,
Extract audit is the other tab).

---

## Appendix G — Side notes

- Site HTML is a visual retelling; this file plus `docs/` remain the
  source of truth for explanations and numbers.
- Free T4 compute shaped real choices (fp16, model size, 5000 steps,
  last checkpoint). More compute would mainly buy longer training and
  a true eval holdout (#73) so “undertrained” can be separated from
  “structurally ungrounded confidence” at a higher accuracy floor.
- Stages 2b (RLVR **policy**), Stage 3 demo curves, and a positive
  control remain open. **Stage 5b and Stage 6 are computed offline**
  (not in the preprint); see Preface and Appendix B.
- Biggest scientific gaps for a follow-up: positive-control
  autoregressive Devanagari reader; genuine held-out synthetic split
  before training; Probe 1 redesign that preserves learnability.

---

## Appendix H — Interview Q&A

Short answers for common attacks. Numbers cite docs; do not invent.

### "Your model barely reads. Why should I care?"

> Accuracy was not the experimental objective. I intentionally trained a
> small model from scratch so I could control its training history and
> inspect internal probability distributions. That made it a **measurement
> instrument**, not a production OCR competitor. Grapheme CER ≈ **0.985**
> on text-bearing vs ≈ **0.949** on blank
> ([`docs/paper_defensibility_stats.md`](./docs/paper_defensibility_stats.md)).

### "Maybe it just wasn't trained enough."

> Legitimate alternative. We ran training-curve snapshots (Probe 3b),
> encoder ablation, unseen-script tests, blank/noise/scramble controls,
> and position-resolved teacher-forced likelihood. Evidence is consistent
> with a confidence peak that is largely decoder-prior driven. I would
> not claim we fully rule out insufficient visual learning — that's why
> a **positive-control reader** under the same position-0 protocol is the
> most important next experiment.

### "Why didn't your exposure experiment work?"

> The exposure **manipulation** worked — realized distributions within
> TV ≤ 0.08 (natural **0**, flat ≈ **0.047**, inv ≈ **0.005**). But
> flattened/inverted models sat at the accuracy floor (**0.33%** /
> **0.67%** line accuracy vs **18.3%** natural —
> [`docs/probe1_fixed_effects.md`](./docs/probe1_fixed_effects.md)), so
> we cannot estimate an exposure coefficient. Likely confound: frequency
> inversion disrupted linguistic regularity. Next: natural-sentence
> reweighting + held-out test + mixed-effects model.

### "Why not just use Sarvam's production model?"

> An API tells us what it outputs, not the counterfactuals we need:
> blank the encoder, inspect the full softmax, measure *p*(GT) at
> position 0, or control training exposure. Sarvam is the **external
> production reference**; the instrument is the mechanistic lab.

### "Sarvam's confidence and your model's confidence are completely different."

> Correct — we do not treat them as numerically comparable. Stage 5 asks
> two **transfer** questions: (5a) does Sarvam's page confidence track
> its own published language-level accuracy spread? (5b) does our
> instrument's per-image difficulty rank predict Sarvam CER? Neither is
> a calibration comparison between models.

### "Why do you need an n-gram model? You already have the OCR model."

> I needed a **language-only baseline**. If the OCR model produces
> plausible text on blank images, I must know whether that matches what
> a zero-visual-input model could generate from statistics alone. At
> position 0 the n-gram prior (−**4.73** nats) massively outscores the
> instrument (−**24.54** nats —
> [`docs/position_matched_ngrams.md`](./docs/position_matched_ngrams.md)).

### "Why is position zero better than later tokens?"

> Later tokens can be predicted from the generated prefix. At position 0
> there is no prefix — only the image can supply input-dependent
> information for the first grapheme. That makes teacher-forced *p*(GT)
> a cleaner grounding diagnostic.

### "Okay position 0 is weird — what happens later?"

> Position 0 is the cleanest case because there is no generated prefix,
> but we measured the **full positional profile**, not just the first
> step. Teacher-forced log *p*(GT) is catastrophic at position 0
> (**−24.54** nats vs a text-only 5-gram's **−4.73** at the first
> grapheme — [`docs/position_matched_ngrams.md`](./docs/position_matched_ngrams.md)),
> still bad at position 1 (**−8.12** vs **−1.14**), then tracks a 4/5-gram
> surprisingly closely on positions 2–39 (within a few tenths of a nat;
> KL **0.27–0.32**; argmax agreement **91–94%** —
> [`docs/ngram_kl_argmax.md`](./docs/ngram_kl_argmax.md)), and diverges
> again at 40+. So the story is not "only the first token is broken" —
> mid-sequence the decoder behaves like a **text-only language model**
> even when the image does not support the output. Position 0 remains the
> killer diagnostic because only the image could have mattered there;
> mid-sequence overlap is what a language-prior account predicts once a
> prefix exists.

### "Why not use max probability as confidence?"

> Max-softmax measures how **peaked** the distribution is around the
> model's preferred answer, not whether that answer is **supported by
> the image**. Our instrument showed ~**0.90** self-confidence while
> *p*(GT) ≈ **2.2×10⁻¹¹** at position 0 and GT was never argmax (0/180).

### "What did you actually contribute?" (Sarvam-style)

> A controlled OCR instrument to decompose confidence into visual
> evidence vs language-prior behavior, with interventions impossible on
> a closed API; then external validation on Sarvam showing no useful
> transfer of our difficulty signal in n=60. Takeaway: production needs
> **grounding-aware confidence**, not raw max-softmax. See **Presenting
> to Sarvam**.

---

*When a deferred stage is built and verified, extend its chapter with
prose and evidence — do not leave the finding only in chat. Flip the
matching row in IMPLEMENTATION.md and append to DECISIONS.md if you
made a new choice.*

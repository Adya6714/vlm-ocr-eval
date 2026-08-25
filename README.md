# vlm-ocr-eval

**A diagnostic instrument for Indic document OCR, not an OCR system.**

This repository asks a narrow question: when a document VLM misreads Devanagari
or Bengali, *why* — and can we tell the difference between a model that cannot
see the glyph and a model that is confidently guessing from its language prior?

Answering that requires a model you own end to end. A closed API returns text;
it does not return the runner-up token distribution, per-step confidence, or
behaviour on a blank image. So this project builds a small instrument model from
scratch, with **zero Indic pretraining exposure**, purely so those diagnostics
are possible — then transfers the findings to a production API.

---

## Status: research apparatus, partially executed

This is an honest description, not a pitch.

| Stage | What it is | Status |
|---|---|---|
| 0 | Error taxonomy over Tesseract / Surya / PaddleOCR | Built, run on real data |
| 1 | Controlled renderer (glyph-frequency dial) | Built, Tier A/B verified |
| 2a | The instrument (from-scratch encoder/decoder) | Built, trained on real Hindi |
| 2b | The demo (LoRA on a pretrained VLM) | **Not built — see Future Work** |
| 3 | Reading-order / table-binding metrics | Not built |
| 4 | Probe suite (1, 2, 3, 5, 6) | Built; 5b not built |
| 5 | Sarvam API transfer | Not built |
| 6 | Triage cascade | Not built |

What exists is real and runs. What doesn't exist is marked clearly. Nothing here
is claimed to be a benchmark result or a dataset contribution.

---

## The core idea

Three things separate this from "run OCR, compute CER":

**1. Not every diff is an error.** Devanagari has many same-sound/different-bytes
encodings — ZWJ/ZWNJ joiners, khanda-ta, anusvara vs. conjunct nasal, danda vs.
full stop, Devanagari vs. Latin digits. Stage 0 builds a two-tier equivalence
check (Tier 1 deterministic encoding equivalence, Tier 2 phonetic equivalence via
ISO 15919 transliteration) and reports what fraction of apparent errors are not
errors at all. On real Tesseract output, roughly one in five apparent errors was a
Tier 1 encoding variant.

**2. Exposure is a dial, not an observation.** Rather than correlating accuracy
against whatever glyph frequencies happen to occur, the renderer *sets* them:
`natural`, `flattened`, and `inverted` glyph-frequency modes, matched on total
data volume, with realized frequencies verified to within TV ≤ 0.08 of target.
Training the same architecture across all three, three seeds each (nine runs),
turns a correlation into something closer to a causal claim.

**3. Confidence is the thing being measured, not accuracy.** A model that fails
loudly is manageable. A model that fails fluently and confidently on a
low-resource script is dangerous. Probe 3 feeds blank and noise-matched images;
Probe 5 buckets predictions by confidence and checks whether confidence tracks
correctness at all.

---

## Findings so far

Early, single-condition, and reported as such.

- **Stage 0:** ~20% of Tesseract's apparent errors on real Hindi were Tier 1
  encoding variants rather than genuine misreads. A large `UNREVIEWED` fraction
  remains — the report is real, not final.
- **Probe 3 (hindi/natural/seed0):** mean confidence was 0.9929 on real line
  crops, 0.9898 on blank images, 0.9877 on matched noise. The model's confidence
  barely distinguishes a real line of text from an empty page.
- **Probe 5 (same checkpoint):** every sampled prediction landed in the top
  confidence bucket (0.9–1.0) while actual accuracy in that bucket was 0.10.

Read together, these say the instrument at this training scale is not grounding
its confidence in the image or in correctness. That is a coherent, expected
result for a small model at 5000 steps on a small corpus — and it is precisely
the failure mode the probe suite exists to detect.

---

## Pipeline

```
Real corpus (GlotOCR)
│
├──► Stage 0 ──► baseline predictions ──► Tier 1/2 equivalence ──► error taxonomy
│
└──► Stage 1 renderer (glyph-frequency dial, HarfBuzz shaping)
       │
       ├──► Tier A (controlled) ──► line-crop manifests ──► Stage 2a training
       ├──► Tier B (degraded)                              │
       └──► Tier C (real docs)                             │
                                                           ▼
                                              9 checkpoints (3 modes x 3 seeds)
                                                           │
       Probe 6 ◄───────────────────────────────────────────┤
                                                           │
       Probes 2, 3, 5 ◄────────────────────────────────────┘
```

---

## Running it

Heavy compute runs on Colab; everything takes a single `--data-root`.

```bash
# render controlled pages and export line-crop manifests
python src/data_pipeline/export_manifest_scaled.py --script hindi --pages-per-mode 100

# train one condition/seed (resumable — assume the session dies)
python src/models/instrument/train.py \
  --manifest data/manifests/hindi_natural.jsonl \
  --output-root <checkpoints> --condition natural --seed 0

# probe it
python src/probes/probe3_blank_control.py --manifest ... --condition natural --seed 0 ...
python src/probes/probe5_calibration.py  --manifest ... --language hindi ...

# aggregate across all nine runs
python src/analysis/aggregate_probe_results.py --script hindi --out data/probe_results/summary_hindi.json
```

`make smoke-test` proves the architecture end to end on synthetic data with no
GPU and no `data/raw` — it produces zero scientific findings, by design.

---

## Documentation map

- `IMPLEMENTATION.md` — technical spec, stage by stage, status-marked
- `DECISIONS.md` — every non-obvious choice, numbered, with the rejected alternative
- `BOOK.md` — first-principles explainer for the CV/RL concepts involved
- `AGENTS.md` — workflow rules for coding agents in this repo
- `COLAB_RUNS.md` — log of what was executed where

`DECISIONS.md` is the file worth reading. It records why grapheme clusters rather
than BPE, why two models rather than one, why three seeds rather than one, why
line-crops are padded to a fixed 70px, and why several appealing ideas were
rejected.

---

## Future work

Deliberately scoped out, with the reasoning rather than just the name.

### 2b — the demo model (LoRA on a pretrained VLM)

The instrument exists to isolate a variable, so it must start blank. A demo model
exists to look like a production system, so it should not. The plan is LoRA
adaptation of a small open document VLM (SmolDocling-256M or LightOnOCR-1B,
chosen on measured T4 memory fit), with **separate** layout and reading-order
modules rather than one monolithic model — mirroring how production document
systems are actually decomposed.

Not built because it is a different project shape: multi-module training,
supervised fine-tuning, then RLVR, none of which is gated on the instrument's
findings. `src/models/demo/benchmark_base_models.py` exists as its one
prerequisite: measure peak VRAM for both candidates under LoRA before choosing.

### 4 — RLVR with a coverage penalty

Reward = character accuracy + structure match (TEDS) + reading-order rank
correlation − a coverage term. The coverage term matters more than it looks:
without it, a model can raise measured accuracy by *saying less*. The single
planned ablation is to remove that term and quantify the model learning to omit
text — a mechanism, not a table of marginal deltas.

### 5 — Sarvam API transfer

The instrument's value is that its per-glyph-class error structure is fully
observable. The open question is whether that structure *rhymes* with a
production system's. Planned as a pre-specified rank-correlation test with a
permutation null, statistic chosen before looking at results, run on both clean
and degraded tiers — clean-only comparison is close to meaningless. Budgeted at
~200 pages, fetched once and cached, with all threshold sweeps run offline
against the cache. A null result is still reported.

### 6 — Triage cascade

Using Probe 5's confidence scores as a router: sweep every escalation threshold
offline against the Stage 5 cache, report accuracy-recovered vs.
fraction-escalated vs. cost-per-page, against three baselines (random,
layout-complexity, Tesseract-confidence). The claim is router *quality*, not
headline cost savings — the instrument is not expected to beat Tesseract on raw
accuracy, and pretending otherwise would be the wrong claim.

### Probe 5b — zero-shot floor

Santhali (Ol Chiki) and Kashmiri (Perso-Arabic) are scripts the instrument has
never seen — not merely under-sampled, but truly absent. If confidence stays high
at genuine zero exposure, that is the sharpest possible version of this
project's central finding.

---

## What more compute would buy

Everything here ran on a free Colab T4. That constraint shaped real decisions —
fp16 rather than bf16 (Turing has no bf16), a ~19M-parameter instrument, 5000
training steps, 100 rendered pages per condition, 30-sample probe runs. The
constraint is stated plainly because it bounds what the results can support.

Concretely, with more compute:

- **Longer training.** The overconfidence in the findings above is consistent
  with undertraining. Ten or fifty times the steps would separate "this model
  hasn't learned to use the image yet" from "this architecture systematically
  fails to ground confidence in vision" — currently those two explanations are
  not distinguishable from the data.
- **A larger corpus.** 100 pages per condition against a 60-line source corpus
  means heavy reuse. More source text would make the glyph-frequency
  manipulation cleaner and reduce memorization.
- **Full probe sampling.** Probes currently sample 30 items per run for time.
  Running the full manifest would tighten every number and make per-glyph-class
  breakdowns (the actual point of Probe 1) statistically meaningful.
- **All four languages.** Santhali and Kashmiri need font support and a
  separate render path; Bengali training is a full nine-run sweep of its own.
- **Stage 2b at all.** LoRA on a 1B VLM plus two auxiliary modules resident
  simultaneously is the specific thing a T4 makes painful.

None of this changes the design. It changes how much weight the numbers can bear,
which is worth being explicit about.

---

## A note on what this is

This repository is a diagnostic apparatus and a worked argument about
methodology. It is not an OCR product, not a benchmark, and not a dataset
contribution. Where a stage is unbuilt it says so; where a number is provisional
it says so; where a claim would outrun the evidence, the claim is not made.

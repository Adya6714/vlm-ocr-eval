# IMPLEMENTATION.md

Stages 2b, 3, 5, and 6 were deliberately not executed in this phase —
see README.md § Future Work for the reasoning behind each.

This is the technical spec for ocr-vlm-eval. Every module below has a goal,
concrete inputs/outputs, a file location, and acceptance criteria. A coding
agent should read this file before writing any code, check the status
markers to see what already exists, implement the next unstarted item, then
flip its status and move on.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done ·
`[!]` blocked, see note.

When a module is written and has been run (not merely authored), mark
it `[x] BUILT — QUEUED` if the only evidence is fake/smoke data, or
`[x] BUILT — VERIFIED` if it was run against real project data. `[x]`
alone on older items means the same as VERIFIED unless a note says
otherwise.

Whenever a status flips to `[x]` or `[!]`, also: (1) update `TODO.md`,
(2) append an entry to `DECISIONS.md` if a non-obvious choice was made
along the way, (3) write or extend the matching chapter in `BOOK.md`.
See `AGENTS.md` for the exact workflow. End-to-end architecture
without real data or a GPU: `make smoke-test` (see `makefile`;
the phony target is currently `smoke-test`).

---

## Stage 0 — Error taxonomy from existing engines (no GPU, no training)

**Goal:** see what Indic OCR failure actually looks like, by hand, before
building anything. Produces the labeled taxonomy that later stages report
against, and produces the encoding-equivalence tables used in Probe 4.

- [~] `src/eval/run_baselines.py` — run Tesseract, Surya, and PaddleOCR
      over a seed set of ~200 real Devanagari + Bengali document images.
      Input: image directory. Output: one JSON per engine per image, with
      raw predicted text and (where the engine exposes it) confidence.
      Engine API regressions fixed 2026-08-24 (DECISIONS.md #28): Surya
      typo `rognition`→`recognition` + 0.22 full-page API; PaddleOCR 3.x
      drop `show_log`, use `predict()`, lang=`hi`. Hardened further
      (DECISIONS.md #42): `use_textline_orientation=False`,
      `enable_mkldnn=False` (CPU OneDNN crash). Smoke 2026-08-24:
      Hindi+Bengali `--limit 2` via `--pred-root` → 4/4 non-null.
      Resumable append+skip + per-image progress (DECISIONS.md #31).
      Colab-facing: `--data-root` / `--export-zip` (DECISIONS.md #32).
      Tesseract+Surya jsonl OK; paddleocr jsonl still mostly null
      `show_log`/OneDNN rows — strip failed paddleocr lines then
      resume `--engine paddleocr` only (do not delete the files).
- [x] BUILT — VERIFIED `src/eval/error_taxonomy.py` — align each
      prediction to ground truth at the grapheme-cluster level (not
      code-point level — see `DECISIONS.md` #7) and bucket every diff
      into: exact match, encoding-variant (Tier 1), phonetic-variant
      (Tier 2), genuine misread / dropped matra/nukta / reading-order
      break / hallucinated-repeated (from hand-review notes), or
      UNREVIEWED. Output: `data/predictions/error_taxonomy.csv` (481
      non-header rows from real engine jsonl) plus a printed per-engine
      fraction table. UNREVIEWED remains large until more
      `hand_review.py` notes exist — the report is real, not final.
- [x] BUILT — VERIFIED `src/eval/equivalence_tables.py` — **Tier 1, encoding equivalence.**
      Deterministic, hand-curated table of same-sound/different-bytes
      variants: ZWJ/ZWNJ joiner variants, khanda-ta vs. ta+hasant+ZWJ,
      anusvara vs. conjunct nasal, danda vs. full stop, Devanagari vs.
      Latin digit systems, common punctuation-normalization pairs (smart
      quotes, hyphens — note olmOCR-bench already NFC-normalizes these,
      see `DECISIONS.md` #4). Output:       a lookup used by the scorer to mark
      a diff as "not an error." `__main__` self-test run (9/9). Applied
      to real predictions inside `error_taxonomy.py`.
- [x] BUILT — VERIFIED `src/eval/transliteration_equivalence.py` — **Tier 2, phonetic
      equivalence.** Wraps a transliteration library (aksharamukha or
      indic_transliteration, target scheme: ISO 15919 — not a lossy
      romanization, see `DECISIONS.md` #8) to canonicalize both reference
      and hypothesis, then compares. Reported separately from Tier 1 —
      this is a judgment call about what counts as "the same word," not
      an uncontroversial encoding fact. Wired into `error_taxonomy.py`.
  - [x] Validation set: 38 hand-checked pairs (23 positive / 15
        negative+boundary); `python3 src/eval/transliteration_equivalence.py`
        → 38/38. Classes: nukta NFC/NFD, virama+ZWJ/ZWNJ, ॐ↔ओं,
        Bengali ৎ↔ত্; negatives per #18. Docs:
        `docs/tier2_validation.md`, DECISIONS.md #54. Honest <40 —
        unverified candidates left as TODO, not invented.
  - [ ] LLM-as-judge spot-check: run Claude over the cases Tier 2 flags
        as _disagreeing_ with a human label, to catch false negatives in
        the transliteration library itself. Validation only — the
        reported metric stays the deterministic transliteration match, not
        the LLM judgment (reproducibility).
        **Note:** corpus TIER2 rate is currently 0% after Tier 1
        (re-ran `error_taxonomy.py` 2026-09-03) — phonetic residuals
        are rare here; LLM spot-check waits until Tier 2 fires on
        real diffs or on the validation-set disagreements.
- [x] `src/eval/hand_review.py` + `src/eval/hand_review_assist.py` —
      interactive hand-reading viewer. Tier 1/2 explained diffs are
      skipped; every UNEXPLAINED engine output gets a *suggested*
      residual label from the fixed set (genuine-misread,
      dropped-matra-nukta, reading-order-break,
      hallucinated-repeated-text) or an explicit no-fit, plus a short
      reason. The human confirms with Enter, overrides by typing, or
      skips — the notes file records
      `agent-suggested-and-confirmed` vs `human-overridden`. Heuristic
      validation: `python3 src/eval/hand_review_assist.py` (13/13,
      includes whitespace-only no-fit pair); note-outcome checks:
      `PYTHONPATH=src/eval python3 src/eval/hand_review.py --self-test`
      (7/7). Applies Tier 0 `normalize_whitespace()` before heuristic
      diffing (DECISIONS.md #22). `--queue PATH` adjudicates a
      pre-drawn sample jsonl (notes schema) and appends to
      `hand_review_notes.jsonl` (DECISIONS.md #55).
- [x] BUILT — VERIFIED `src/analysis/adjudication_sample.py` —
      stratified UNREVIEWED sample (n=200, seed=42) →
      `data/predictions/adjudication_sample.jsonl`; bootstrap Tier 1
      rate over full non-exact population (CI withheld until humans
      label); ranking test raw vs Tier-1 error rates. Report:
      `docs/adjudication_analysis.md`. Ranking stable on current
      corpus (engines/languages not reordered).

**Acceptance:** a report stating, per engine, what fraction of all
reported errors are Tier 1, what fraction are Tier 2, and what fraction
are genuine. This report is the input to the interview's strongest single
claim, so it needs to be reproducible from a script, not from memory.
The 20.4% Tier-1-among-non-exact headline stays provisional until the
adjudication sample is labeled and the bootstrap CI is reported
(DECISIONS.md #55).

---

## Stage 1 — The renderer

**Goal:** a data engine that takes a glyph-frequency distribution as a
parameter. This is what makes Stage 5 / Probe 1 possible — it is
experimental apparatus, not a utility.

- [~] `src/renderer/layout_sources.py` — pulls real document layouts
      rather than inventing templates: Internet Archive scanned Indic
      books, India.gov.in multilingual PDFs, Wikipedia Indic-language
      articles rendered as realistic web-doc layouts. See
      `DECISIONS.md` #9. Output: a layout template bank (single-column,
      two-column, marginalia, table-embedded, form).
      **PARTIAL (audit 2026-08-24):** extractors are real, but
      `bank.json` was missing at audit; india.gov/NCERT PDF fetch
      fails; bank lacks `form` / `table-embedded`; deferred.
- [~] `src/renderer/degradation_profile.py` — measures blur, noise,
      skew, and show-through parameters off ~20 real scanned pages
      (prescriptions, photocopied forms, a textbook, a paper with
      tables/graphs) and stores them as a sampleable distribution, not
      hardcoded constants.
      **PARTIAL (audit 2026-08-24):** parameters are measured (not
      hardcoded), but the pool is IA book pages + GlotOCR synthetic
      degraded renders — not prescriptions/photocopied forms. Deferred.
- [x] `src/renderer/glyph_frequency.py` — the core knob. Given a corpus
      and a target distribution mode (`natural` / `flattened` / `inverted`),
      resamples/synthesizes source text so realized glyph-cluster
      frequencies match the target within `TARGET_TV_TOLERANCE` (TV ≤
      0.08; see acceptance below and `DECISIONS.md` #28), while keeping
      local language-model plausibility as close to natural as the mode
      allows via bigram-guided packing (see `DECISIONS.md` #10).
- [x] `src/renderer/render.py` — HarfBuzz-backed text shaping (do not
      hand-roll conjunct placement — Indic shaping is genuinely hard, see
      `BOOK.md` Chapter 2), multiple fonts per script, cycles through
      layout templates, applies a sampled degradation profile. Output: a
      page image plus exact per-character/per-grapheme-cluster ground
      truth with bounding boxes.
- [x] Three realism tiers, all built on the same renderer:
  - [x] Tier A — controlled: fixed fonts, fixed degradation, only
        glyph frequency varies. Used for Probe 1's causal validity.
  - [x] Tier B — degraded: parameters sampled from real scans (above).
        Used for headroom in Probes 2–5.
        (Audit note: Tier B works and differs from A; degradation
        *source quality* still inherits the PARTIAL profile above.)
  - [x] Tier C — real: unmodified real documents where ground truth
        already exists. Used as the reality check (Stage 5, Probe 6).

**Acceptance:** given a target script, layout, and glyph-frequency mode,
`render.py` produces an image + ground-truth pair in under 1 second, and
a histogram check confirms realized frequencies match the target
distribution within tolerance. **Tolerance (explicit):** total-variation
distance TV(realized, target) ≤ 0.08 for every mode on the Probe-1
corpus (`TARGET_TV_TOLERANCE` in `src/renderer/glyph_frequency.py`).
Natural must be ≈ 0 (exact multiset). Verified on the 60-line Hindi
GlotOCR slice after #28.

---

## Stage 2 — The two models

### 2a. The instrument (from scratch)

- [x] BUILT — VERIFIED `src/models/instrument/tokenizer.py` — grapheme-cluster
      vocabulary, not BPE (see `DECISIONS.md` #2). Built from the
      training corpus, not a frozen universal vocab. Trained on real
      Hindi line manifests (Probe 1 × 9 runs); `__main__` round-trip
      smoke also passes on fake strings.
- [x] BUILT — VERIFIED `src/models/instrument/encoder.py` — small ViT, trained from
      scratch, no pretrained weights (zero pretraining exposure is the
      whole point). Real Hindi training produced the 9 checkpoints used
      by Probe 3/5 jsonl under `data/probe_results/`.
- [x] BUILT — VERIFIED `src/models/instrument/decoder.py` — 4–6 layer autoregressive
      decoder over the grapheme-cluster vocabulary, cross-attending to
      encoder features. Same real-Hindi verification as encoder; smoke
      via `make stage2-instrument-smoke`.
- [x] BUILT — VERIFIED `src/models/instrument/train.py` — line-level training first
      (fast iteration), then page-level. fp16, gradient checkpointing
      (Turing T4 has no bf16 — see `DECISIONS.md` #3 / hard constraints).
      Checkpointing must be resumable mid-run; assume the Colab session
      dies. Consumes a line-crop JSONL manifest `{"image_path","text"}`
      (DECISIONS.md #37). Verified: 9 Hindi Probe 1 checkpoints
      (3 conditions × 3 seeds) trained on real manifests; evidence is
      the Probe 3/5 result files (checkpoints live on Colab Drive).
- [x] BUILT — VERIFIED `src/models/instrument/generate.py` — greedy decode
      (no beam, no KV cache) returning text, token ids, per-step
      confidence, and top-k — what Probes 2/3/5 need. Verified via
      Probe 3/5/5b inference on real Hindi checkpoints.
- [x] BUILT — QUEUED `scripts/make_fake_probe1_data.py` — noise line-crops
      + manifest so the instrument/Probe 1 machinery can run without
      Stage 1 renderer output. Not a finding. Remains QUEUED: this
      *is* the fake path.
- [x] Target size: ~30–60M params (~19.5M realized). Trains on one
      script in well under an hour on a free T4 — verified on real
      Hindi manifests (9 Probe 1 runs).

### 2b. The demo (LoRA on a real small VLM)

- [ ] `src/models/demo/base_model.py` — loads SmolDocling-256M or
      LightOnOCR-1B (decide per `DECISIONS.md` #3 once both are
      benchmarked for T4 memory fit).
- [ ] `src/models/demo/lora_config.py` — LoRA adapter config.
- [ ] `src/models/demo/layout_module.py` — separate small detector for
      block-level layout, trained on renderer ground truth.
- [ ] `src/models/demo/reading_order_module.py` — separate module,
      pointer-network or pairwise-relation style, scored with Kendall tau
      (not accuracy — ordering is a permutation problem).
- [ ] `src/models/demo/sft.py` — supervised fine-tune on Tier A/B
      renderer output with exact ground truth.
- [ ] `src/models/demo/rlvr.py` — reward = character accuracy +
      structure match (TEDS) + reading-order rank correlation − a
      coverage term (penalizes omitting text, so the model can't game
      accuracy by saying less).
  - [ ] Coverage-term ablation: retrain with the coverage term removed,
        confirm and quantify the model learning to omit text. This is a
        two-hour experiment and a specific, checkable claim — keep it as
        the _only_ RLVR ablation (see `DECISIONS.md` #11 on scope).

**Acceptance:** the instrument trains three times (Stage 5, Probe 1)
without manual intervention and produces per-glyph-class accuracy and
confidence. The demo produces markdown/HTML output with layout tags and
an ordered block sequence, comparable in structure to Sarvam's Digitise
output.

---

## Stage 3 — Structure metrics

- [ ] `src/eval/reading_order_metric.py` — Kendall tau between
      predicted and ground-truth block order, computed per layout-
      complexity bucket (single-column → two-column → marginalia →
      table-embedded), so degradation-vs-complexity is a curve, not one
      number.
- [ ] `src/eval/table_binding.py` — the scoped-down table-to-prose idea
      (see `DECISIONS.md` #12): after OCR, can each cell still be bound to
      its correct column header? One number per table, clean ground truth
      from the renderer. Not full prose generation.

**Acceptance:** a tau-vs-complexity curve for the demo model, and a
binding-accuracy number comparable against Sarvam's Extract endpoint
(which returns per-field confidence — see `DECISIONS.md` #13).

---

## Stage 4 — The probe suite (the actual deliverable)

All six probes run on **the instrument**, except Probe 4 (Tier 1/2
equivalence, already built in Stage 0). Probe 6 compares synthetic
Claim B numbers to real Tier C (paper scope; DECISIONS.md #58).

- [x] BUILT — QUEUED **Probe 1 — Exposure vs. complexity (orchestrator only).**
      `src/probes/probe1_exposure.py` trains the instrument three times
      (`natural` / `flattened` / `inverted`), 3 seeds each (9 runs —
      DECISIONS.md #14), identical data volume. In-process; skip-if
      checkpoint already at `total_steps` (DECISIONS.md #39). Fake-data
      path: `scripts/make_fake_probe1_data.py` + `make probe1-smoke`
      (three condition-specific manifests; wiring fixed 645ae11).
      Fake data carries no exposure signal. Real Probe 1 needs the 9
      Colab runs on `data/manifests/{hindi,bengali}_*.jsonl` (manifests
      exist; FE fit still TODO). `make smoke-test` is the no-GPU
      architecture proof and passes.
  - [x] Fit per-glyph-cluster accuracy against log exposure with glyph
        fixed effects; complexity = glyph fixed effect (estimated, not
        residual). **Ran on Hindi Colab artifacts — headline β withheld:**
        flattened/inverted line accuracy ~0% makes FE fit uninterpretable.
        `src/analysis/probe1_fixed_effects.py` → `docs/probe1_fixed_effects.md`
- [x] BUILT — VERIFIED **Probe 2 — Confusion structure.** GT-aligned
      grapheme substitutions on Probe 5b's Hindi Tier C sample
      (`Random(0)`, pool=60; seeds 0–2). For each substitution we record
      true cluster, predicted cluster, and the runner-up structure from
      the model’s full softmax (including correct-cluster p(true) and
      rank). This produces a confusion graph over glyph classes that
      can’t be reproduced from a closed OCR API.
      Outputs: `data/probe_results/probe2_hindi_natural_seed{0,1,2}.jsonl`;
      analysis: `docs/probe2_confusion_analysis.md`. Unit tests 10/10.
- [x] BUILT — VERIFIED **Probe 3 — Reading vs. guessing.** Feed blank/noise
      images. Whatever accuracy survives is language-model guessing, not
      vision. Decompose per glyph class; cross-reference against Probe 1's
      exposure levels — this is the mechanistic account of why low-exposure
      languages hallucinate fluently instead of failing loudly.
      `src/probes/probe3_blank_control.py` — 9 files, n=100
      (`data/probe_results/probe3_hindi_{natural,flattened,inverted}_seed{0,1,2}.jsonl`).
- [x] BUILT — VERIFIED **Probe 3b — Training-curve disambiguation.** Re-run
      Probe 3's real-vs-blank confidence comparison at multiple training
      steps to separate "confidence ungrounded in the image" from "simply
      undertrained." Requires `train.py --keep-snapshots` on a fresh run
      (default resume checkpoint overwrites intermediates).
      `src/probes/probe3_training_curve.py` — 5 snapshots (steps
      500/1000/2000/3000/5000) on hindi/natural seeds 0–2
      (`data/probe_results/probe3_curve_hindi_natural_seed{0,1,2}.json`).
      Analysis: `src/analysis/analyze_probe3_curve.py` →
      `docs/probe3_curve_analysis.md`. Gap sign-flips 4/5 steps /
      |SD|>|mean| 4/5 → indistinguishable from zero across seeds;
      step-3000 all-negative (mean −0.0075) noted separately —
      `docs/statistical_repair.md`, DECISIONS.md #53.
- [ ] **Probe 4 — Encoding/phonetic equivalence.** Already built,
      Stage 0. Re-run here against the instrument's own outputs.
- [x] BUILT — VERIFIED **Probe 5 — Calibration under exposure.** Does
      confidence predict correctness, and does calibration break down
      specifically for the glyph classes starved in Probe 1's `inverted`
      condition? This is Probe 1 × Probe 5 crossed and is the project's
      centerpiece finding — flag it as such in any write-up.
      `src/probes/probe5_calibration.py` — 9 files, n=100
      (`data/probe_results/probe5_hindi_{natural,flattened,inverted}_seed{0,1,2}.jsonl`).
- [x] BUILT — VERIFIED **Probe 5b — Zero-shot floor.** Santhali (Ol Chiki)
      and Kashmiri (Perso-Arabic) — scripts the instrument has _never_
      seen, not just under-sampled. Check whether confidence collapses
      correctly at true zero exposure, or stays falsely high the way
      GlotOCR Bench found production models doing. No training needed —
      inference only. `src/probes/probe5b_zeroshot_floor.py` — run on
      hindi/natural seeds 0–2, 720 records across
      hindi/santhali/kashmiri/blank
      (`data/probe_results/probe5b_hindi_natural_seed{0,1,2}.jsonl`).
      Analysis: `src/analysis/analyze_probe5b.py` →
      `docs/probe5b_analysis.md` (across-seed mean±SD; between-cond
      range 0.0037 < seed SDs; Kashmiri Bonferroni retracted;
      script substitution 360/360; TOST δ=0.05;
      `docs/statistical_repair.md`, DECISIONS.md #52–#53).
- [x] BUILT — VERIFIED **Attention ablation (Claim B mechanism).** Encoder
      memory is ablated by zeroing its output before `memory_projection`
      (prior-only condition), then decoding with greedy + teacher-forced
      shared prefixes. Pooled over 3 Hindi/natural seeds on the same
      Probe 5b Tier C sample: mean confidence is nearly unchanged
      (full 0.9861 vs zero-memory 0.9891; delta −0.0030), but the
      token distributions differ (top-1 agreement 0.8794, prior
      sufficiency 0.8827, mean KL(full||zero) 1.075). Interpretation:
      confidence is close to prior-dominated, while only ~12% of token
      choice mass depends on image content (DECISIONS.md #56).
      Outputs: `data/probe_results/attention_ablation_hindi_natural_seed{N}.jsonl`;
      analysis: `docs/attention_ablation_analysis.md`. Unit tests 11/11.
- [x] BUILT — VERIFIED **Probe 6 — Synthetic-to-real gap (paper scope).**
      Tier C Hindi plain+degraded+blank vs synthetic Claim B (Probe 3/5).
      Held-out validity is leakage-free (0 overlaps between
      `data/manifests/hindi_*.jsonl` and `data/raw/hindi/images/`). Pooled
      over 3 seeds on the same real sample:
      mean confidence real_plain 0.9861, real_degraded 0.9768, blank 0.9799
      while Tier 1/2 line accuracy is 0.0 across the scored real sample.
      Outputs: `data/probe_results/probe6_synthetic_real_hindi_seed{N}.jsonl`;
      analysis: `docs/probe6_synthetic_real_analysis.md`. Unit tests 5/5.

**Acceptance:** each probe produces one clear plot/table and one sentence
that states a finding and its implied fix (see `BOOK.md` for why "implied
fix" is the bar, not just "here's a number").

---

## Stage 5 — Sarvam transfer

**Budget: ~200 pages total, ₹0.5/page, max 10 pages/job (confirmed
docs.sarvam.ai, Sept 2026).  Allocate up front, do not spend ad hoc.**

### Stage 5a — minimal confidence-gap probe (BUILT — ready to run)

- [x] `src/eval/sarvam_client.py` — thin wrapper over Sarvam Doc-AI
      **Extract** endpoint (not Digitise — only Extract exposes
      `annotations.{field}.confidence`, confirmed Sept 2026).  Uses a
      trivial single-field schema (`{"full_text": "..."}`) to yield one
      confidence number per page comparable to `mean_confidence`.
      **Every page cached once** to `data/cache/sarvam/` by SHA-256 of
      image bytes; re-running never re-calls the API (DECISIONS.md #19).
      Handles full async lifecycle: POST extract → poll
      `/doc-ai/v1/job/{job_id}/status` → GET
      `/doc-ai/v1/job/{job_id}/results` → extract
      `annotations.full_text.confidence`.  SARVAM_API_KEY read from env,
      never logged. API schema confirmed from live docs Sept 2026.

- [x] `src/probes/sarvam_transfer_probe.py` — **BUILT — VERIFIED (RUN).**
      35-image sample (10 Hindi / 10 Santhali / 10 Kashmiri / 5 blank)
      drawn with `Random(0)`. Budget used: **₹17.50** (35 pages × ₹0.5).
      Results cached in `data/cache/sarvam/` (35 JSON files).
      Output: `data/probe_results/sarvam_transfer_probe.jsonl`.
      Analysis: `docs/sarvam_transfer_analysis.md`.

      **Headline finding:** Sarvam's confidence does NOT track its own
      published accuracy gap. Hindi → Kashmiri accuracy gap: **39.98 pp**
      (95.91 → 55.93); Hindi → Kashmiri confidence delta: **0.0027**
      (0.9997 → 0.9970). Blank images score 0.0000.
      This replicates the instrument's own Claim B finding on a
      production-quality 3B-parameter system.

      Published accuracy used (verified sarvam.ai/blogs/sarvam-vision,
      Sept 2026; DECISIONS.md #60):
        Hindi 95.91 %, Santhali 80.32 %, Kashmiri 55.93 %

- [ ] `src/analysis/analyze_sarvam_transfer.py` — bootstrap CIs +
      full comparison table. **Not yet built** (Stage 5b, deferred).

### Stage 5b — full transfer analysis (DEFERRED)

The original Stage 5 spec also called for:
- Rank-correlation test (with permutation null) between the instrument's
  per-glyph-class error rates and Sarvam's, on Tier A (clean) and Tier B
  (degraded) — see DECISIONS.md #15.
- `src/eval/transfer_analysis.py` with pre-specified statistic.
- 60 + 40 pages for core probe + degradation conditions, 60 for cascade.

This fuller version is deferred — run Stage 5a first and commit results
before spending more budget.

**Acceptance (5a):** 35 cached JSON records in `data/cache/sarvam/`,
one output JSONL, and an honest report of whether Sarvam confidence tracks
its own per-language accuracy gap.

---

## Stage 6 — Triage cascade demo

- [ ] `src/probes/cascade.py` — using Probe 5's confidence scores, sweep
      every escalation threshold **offline against the Stage 5 cache**.
      Report accuracy-recovered vs. fraction-escalated vs. cost-per-page.
      Compare against three baselines: random escalation, layout-
      complexity escalation, Tesseract-confidence escalation — the point
      is router _quality_, not headline cost savings, because the
      instrument is not expected to be more accurate than Tesseract (see
      `DECISIONS.md` #16).

**Acceptance:** one accuracy-vs-escalation-rate curve, instrument vs.
three baselines.

---

## Tooling (not in the original spec)

- [x] BUILT — VERIFIED `makefile` — `make smoke-test` is the architecture
      proof: Stage 0 tier self-tests, instrument `__main__` smokes, all 9
      Probe 1 runs on fake line-crops (`scripts/make_fake_probe1_data.py`
      → `src/probes/probe1_exposure.py`), then `generate.py`. No GPU, no
      `data/raw`. Path/manifest bugs fixed in 645ae11; re-verified
      passing. Same job as `make smoke-test` in README.

---

## Cross-cutting requirements (apply to every stage)

- Every function gets a docstring explaining _why_ it exists and where
  it sits in the pipeline, not just what it does — see `AGENTS.md`.
- No fabricated Sarvam facts. Anything about Sarvam's current API,
  pricing, or published numbers gets pulled fresh (docs.sarvam.ai,
  their GitHub, their blog), not recalled from a stale prior
  conversation — Sarvam ships fast and this project's original
  research pass is already known to be several months stale.
- Every Colab-facing script assumes the session can die mid-run.
  Checkpoint and make resumable by default, not as an afterthought.
  Heavy scripts (OCR batches, training) also follow AGENTS.md "Heavy
  scripts run on Colab": one `--data-root`, no local-only paths, and
  an export step that unpacks into the IMPLEMENTATION.md output path
  for that stage.

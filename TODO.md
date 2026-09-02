# TODO.md

Stages 2b, 3, 5, and 6 were deliberately not executed in this phase —
see README.md § Future Work for the reasoning behind each.

Sequenced task list. Work top to bottom within a stage; stages are mostly
sequential but Stage 0 has zero dependency on anything and should happen
first regardless. See `IMPLEMENTATION.md` for the full spec behind each
item — this file is the ordering and pacing, that file is the detail.

Target pace: roughly one stage per week, six weeks total, with Stage 0
in the first 2-3 days. This will slip. Stages 0, 1, 2, and 4 are the
project — protect time for those first if something has to give.

---

## Stage 0 — Error taxonomy (days 1–3, no GPU)

- [ ] Get ~200 real Devanagari + Bengali document images (scanned books,
  government PDFs, anything real — not rendered text yet)
- [~] Install and run Tesseract, Surya, PaddleOCR over all of them
      (`run_baselines.py` is resumable: append+skip + per-image
      progress; also has per-image hard timeouts so one stuck image
      can't block the batch; DECISIONS.md #31 / #33 / #34 / #42).
      Tesseract + Surya complete. PaddleOCR API fixed (no `show_log`;
      MKLDNN off); strip failed paddleocr jsonl rows then resume
      `--engine paddleocr` only.
- [ ] Read the diffs by hand for at least an hour before writing any
  taxonomy code — this step is the actual point, don't skip to
  automation
- [x] Constrained suggestions on UNEXPLAINED hand-review cases
  (`hand_review_assist.py`: four residual labels or explicit no-fit;
  confirm / override / skip recorded in the notes file)
- [x] Build Tier 1 equivalence table (encoding variants) —
  `equivalence_tables.py`; verified on self-tests + real taxonomy pass
- [x] Build Tier 2 transliteration equivalence (ISO 15919 target) —
  `transliteration_equivalence.py` wired into the scorer
- [ ] Expand Tier 2 validation to ~40 hand-picked pairs (still a seed
  set of mostly negative examples)
- [ ] LLM-as-judge spot-check over Tier 2 disagreements
- [x] First taxonomy report from real predictions
  (`data/predictions/error_taxonomy.csv`)
- [ ] Close UNREVIEWED rows via more `hand_review.py` notes, then
  treat the per-engine Tier 1 / Tier 2 / genuine fractions as final

## Stage 1 — Renderer (days 4–8)

- [~] Source real layouts (Internet Archive, India.gov.in, Wikipedia
  Indic articles) — PARTIAL: extractors exist; bank/india.gov/form/
  table-embedded gaps deferred
- [~] Scan or source ~20 real degraded pages, measure blur/noise/
  skew/show-through parameters — PARTIAL: measured, but pool is IA +
  GlotOCR synthetic degraded, not prescriptions/forms; deferred
- [x] Build glyph-frequency resampling (natural/flattened/inverted) —
  synthesis hits TV ≤ 0.08 (DECISIONS.md #29)
- [x] Build the HarfBuzz-backed render pipeline, ground truth output
- [x] Verify: render 100 pages per tier, spot-check by eye, histogram
  check on realized glyph frequencies (histogram re-verified after #29)

## Stage 2 — The two models (days 9–16)

- [x] Instrument code: tokenizer, encoder, decoder, training loop,
  resumable checkpointing, `generate.py` — architecture smoke via
  `make smoke-test` (verified passing after 645ae11)
- [ ] Instrument: train once on Devanagari at natural frequency, confirm
  it converges at all before anything else (Colab; status TBD)
- [x] Line-crop manifests for Probe 1 / `train.py`: adapter
  `export_line_manifest.py` + `export_manifest_scaled.py`
  (DECISIONS.md #41 / #43 / #44). Full-scale Hindi + Bengali
  `--pages-per-mode 100` lands under `data/manifests/`.
- Stage 3 (demo metrics) remains blocked on Stage 1 layout-bank gaps
  (form / table-embedded / india.gov).
- [ ] Demo: benchmark SmolDocling-256M vs LightOnOCR-1B for T4 memory
  fit, decide, log in DECISIONS.md
- [ ] Demo: LoRA config, SFT run on Tier A/B renderer output
- [ ] Demo: layout module
- [ ] Demo: reading-order module

## Stage 3 — Structure metrics (days 17–21)

- [ ] Kendall tau metric, per layout-complexity bucket
- [ ] Table header-cell binding accuracy metric
- [ ] Run both on the demo model's output

**Blocked on:** Stage 1's real layout bank (`bank.json` exists but is
PARTIAL: missing `form` / `table-embedded`; india.gov fetch flaky).
Tau-vs-complexity needs those buckets populated from real layouts,
not invented templates.

## Stage 4 — SFT then RLVR (days 22–25)

- [ ] RLVR reward function (accuracy + TEDS + rank correlation −
  coverage)
- [ ] Full RLVR training run
- [ ] Coverage-term-removed ablation, confirm and quantify the omission
  failure mode

## Stage 5 — The probe suite (days 26–33)

- [x] Probe 1 orchestrator (`src/probes/probe1_exposure.py`) — 9-run
  loop + skip-if-complete; fake-data path via `make probe1-smoke`
- [x] Fake Probe 1 data + makefile wiring — repaired (645ae11);
  `make smoke-test` passes end to end (no GPU, no `data/raw`)
- [ ] Probe 1: 9 training runs on **real** renderer manifests (3
  conditions × 3 seeds), then glyph-level fixed-effects fit — **ran**;
  headline β withheld (flattened/inverted ~0% line acc); see
  `docs/probe1_fixed_effects.md`
  — Colab; last known mid-inverted seed0; probing (`probe_all.sh`)
  not yet confirmed
- [x] Probe 2 code (`probe2_confusion_graph.py`) — built; needs real ckpts
- [x] Probe 3 code (`probe3_blank_control.py`) — built; needs real ckpts
- [ ] Probe 4: re-run Stage 0's equivalence tables against instrument
  output (Stage 0 scorers already cover the method)
- [x] Probe 5 code (`probe5_calibration.py`) + aggregator — built;
  aggregated calibration table still blocked on Colab probe runs
- [ ] Probe 5b: render Santhali + Kashmiri, zero-shot inference,
  calibration check at true zero exposure (optional / lowest priority)
- [x] Probe 6 code (`probe6_synthetic_real_gap.py`) — built; still needs
  held-out pages 100–109 rendered + instrument/baseline predictions

## Stage 6 — Sarvam transfer + triage cascade (days 34–36)

- [x] **Before spending any budget:** re-verify Sarvam's current
  per-language OCR numbers — done 2026-08-25 (DECISIONS.md #6 Verified:
  Kashmiri 55.93 / Santhali 80.32 still match sarvam.ai Vision blog;
  spread holds).
- [ ] Allocate the ~200-page budget per IMPLEMENTATION.md Stage 5
- [ ] Fetch, cache, never re-fetch
- [ ] Transfer analysis: pre-specify the rank-correlation statistic
  before looking at results
- [ ] Cascade: sweep thresholds offline against cache, compare to the
  three baselines (random, layout-complexity, Tesseract-confidence)

## Ongoing, throughout

- [x] `BOOK.md` teaching book written (rebuild narrative + App. E
  reproduce commands); keep current as new verified results land
- [ ] Methodology upgrades (DECISIONS.md #46 / BOOK after Conclusion):
  equal-frequency Probe 5 bins + ECE/Brier; Probe 3 attention +
  patch-shuffle; 3-seed mean±std; mixed-effects Probe 1; kappa on a
  hand-review subsample; bootstrap CIs — not this phase
- [ ] Keep `DECISIONS.md` current — append, don't rewrite
- [x] Heavy scripts (OCR batches, training) written for Colab: one
  `--data-root`, no local-only paths, export into the IMPLEMENTATION.md
  output path (`run_baselines.py` + AGENTS.md; DECISIONS.md #32)
- [x] `Makefile` / `make smoke-test` — live architecture proof on fake
  data (fake manifests + `src/probes/probe1_exposure.py` path fixed
  in 645ae11; re-verified passing)
- [ ] Before the interview: reread the whole `README.md` framing
  sentence out loud, make sure it still matches what actually got built

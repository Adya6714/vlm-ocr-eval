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
  resumable checkpointing, `generate.py` — smoke-only (`make
  stage2-instrument-smoke` / `make smoke-test`)
- [ ] Instrument: train once on Devanagari at natural frequency, confirm
  it converges at all before anything else
- **Blocked:** `train.py` wants line-crop manifests `{"image_path","text"}`.
  Tier A/B `PageGT` now includes `lines[]` with page-space boxes
  (DECISIONS.md #41). Adapter exists: `export_line_manifest.py` +
  batch driver `export_manifest_scaled.py` (DECISIONS.md #43; hindi
  smoke 3 pages/mode verified). Full-scale Hindi/Bengali export
  (`--pages-per-mode 100`) still pending before real instrument train.
  Stage 3 (demo metrics) is separately blocked on the Stage 1
  layout-bank gaps (form / table-embedded / india.gov).
- [ ] Demo: benchmark SmolDocling-256M vs LightOnOCR-1B for T4 memory
  fit, decide, log in DECISIONS.md
- [ ] Demo: LoRA config, SFT run on Tier A/B renderer output
- [ ] Demo: layout module
- [ ] Demo: reading-order module

## Stage 3 — Structure metrics (days 17–21)

- [ ] Kendall tau metric, per layout-complexity bucket
- [ ] Table header-cell binding accuracy metric
- [ ] Run both on the demo model's output

**Blocked on:** Stage 1's real layout bank (`layout_sources.py` still
PARTIAL: no `bank.json` at last audit; missing `form` / `table-embedded`;
india.gov fetch fails). Tau-vs-complexity needs those buckets populated
from real layouts, not invented templates.

## Stage 4 — SFT then RLVR (days 22–25)

- [ ] RLVR reward function (accuracy + TEDS + rank correlation −
  coverage)
- [ ] Full RLVR training run
- [ ] Coverage-term-removed ablation, confirm and quantify the omission
  failure mode

## Stage 5 — The probe suite (days 26–33)

- [x] Probe 1 orchestrator (`src/probes/probe1_exposure.py`) — 9-run
  loop + skip-if-complete; fake-data only
- [ ] Probe 1: 9 training runs on **real** renderer manifests (3
  conditions × 3 seeds), then glyph-level fixed-effects fit
- **Blocked:** real Probe 1 waits on the Stage 1 line-crop adapter
  above. `make probe1-smoke` currently looks for `probe1_exposure.py`
  under `src/models/instrument/` (file is in `src/probes/`).
- **Blocked:** `scripts/make_fake_probe1_data.py` is encoding-corrupted
  (mangled `FAKE_TEXTS` / missing `build_fake_manifest` header) — smoke
  path will not run until that script is repaired.
- [ ] Probe 2: confusion graph from output distributions
- [ ] Probe 3: blank/noise-image control, per glyph class
- [ ] Probe 4: re-run Stage 0's equivalence tables against instrument
  output
- [ ] Probe 5: calibration curve, cross-referenced against Probe 1's
  exposure levels — this is the centerpiece, budget real time for it
- [ ] Probe 5b: render Santhali + Kashmiri, zero-shot inference,
  calibration check at true zero exposure
- [ ] Probe 6: Tier C real documents + handwriting anecdote (15-20
  lines, 2-3 writers) vs Tier A/B synthetic

## Stage 6 — Sarvam transfer + triage cascade (days 34–36)

- [ ] **Before spending any budget:** re-verify Sarvam's current
  per-language OCR numbers against docs.sarvam.ai / their latest blog —
  the original research pass is from February and Sarvam Vision 1.5 has
  since shipped. Confirm the motivating 40-point spread still holds
  before building the interview narrative on it.
- [ ] Allocate the ~200-page budget per IMPLEMENTATION.md Stage 5
- [ ] Fetch, cache, never re-fetch
- [ ] Transfer analysis: pre-specify the rank-correlation statistic
  before looking at results
- [ ] Cascade: sweep thresholds offline against cache, compare to the
  three baselines (random, layout-complexity, Tesseract-confidence)

## Ongoing, throughout

- [ ] Keep `BOOK.md` current — a chapter per completed stage/probe,
  written close to when the work happens, not batched at the end
- [ ] Keep `DECISIONS.md` current — append, don't rewrite
- [x] Heavy scripts (OCR batches, training) written for Colab: one
  `--data-root`, no local-only paths, export into the IMPLEMENTATION.md
  output path (`run_baselines.py` + AGENTS.md; DECISIONS.md #32)
- [x] `Makefile` / `make smoke-test` — intended no-data architecture
  proof (blocked on fake-data script + Probe 1 path mismatch above)
- [ ] Before the interview: reread the whole `README.md` framing
  sentence out loud, make sure it still matches what actually got built

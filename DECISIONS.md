# DECISIONS.md

A running log of _why_, not just _what_. Every non-obvious choice gets an
entry here: the decision, what else was considered, and the reasoning.
When code looks like an odd choice, this is where the answer lives.

Agents: append to this file (don't rewrite past entries) whenever you make
a choice that isn't the obvious/only option. Number entries sequentially.
Date format: YYYY-MM-DD.

---

### 1. Two separate models — "the instrument" and "the demo" — instead of one

**Decision:** build a small from-scratch model with zero Indic
pretraining exposure (the instrument, used for Probes 1/2/3/5) and,
separately, a LoRA-adapted pretrained small VLM with Sarvam's component
shape (the demo, used to prove production architecture).

**Alternatives considered:** one LoRA-finetuned model doing double duty
as both the causal-exposure apparatus and the production demo.

**Why rejected:** any available small open VLM (SmolDocling, LightOnOCR,
even Sarvam-1 at 2B) has already seen a large volume of Indic text during
pretraining. If Probe 1 varies _fine-tuning_ mixture on top of that, the
exposure manipulation is a rounding error against pretraining exposure,
and the causal claim collapses on first serious question. The instrument
has to start blank. The demo doesn't need to — it exists to look like a
real system, not to isolate a variable.

---

### 2. Grapheme-cluster vocabulary for the instrument, not BPE

**Decision:** the instrument's output vocabulary is grapheme clusters
(the _akshara_ unit — base consonant + matras + virama + ZWJ, as a single
token), not byte-pair-encoded subwords.

**Alternatives considered:** reuse a standard BPE tokenizer (e.g.
Sarvam's own, or a generic multilingual one).

**Why:** Probe 1 needs to measure exposure _per visual unit_. BPE merges
are frequency-driven and would tangle "how often this glyph cluster was
shown to the vision encoder" with "how the tokenizer happened to
segment it" — two different notions of frequency. Grapheme clusters keep
the exposure variable clean and interpretable, and they're also the
correct level for CER in Indic scripts generally (code-point-level CER
understates structural errors — see decision #7).

---

### 3. Base model choice for the demo: TBD, decide once memory-benchmarked

**Decision:** default to whichever of SmolDocling-256M or LightOnOCR-1B
fits comfortably in T4 memory with LoRA + activations + a batch size that
doesn't make training glacial. Not yet benchmarked.

**Why left open:** both are plausible; the actual constraint is T4 VRAM
headroom after the layout and reading-order modules are also resident.
Benchmark both empirically in Stage 2 rather than guessing.

---

### 4. olmOCR-bench's existing NFC normalization

**Decision:** do not re-solve NFC/NFD, hyphen, or smart-quote
normalization — olmOCR-bench (and by extension Sarvam's eval wrapper
around it) already does this before scoring.

**Why it matters:** the project's normalization-artifact claim (Probe 4 /
Stage 0) is _not_ "nobody normalizes Unicode." It's specifically about
the equivalence classes NFC does not cover — ZWJ/ZWNJ variants, anusvara
vs. conjunct nasal, khanda-ta encodings, danda vs. full stop, digit
systems — which are real orthographic ambiguities that NFC, by design,
has no opinion on. Getting this distinction right is what keeps the
claim defensible instead of embarrassing.

---

### 5. (reserved — renumber if inserted retroactively)

---

### 6. Script scope: Devanagari deep, Bengali structural, Santhali/Kashmiri zero-shot floor only

**Decision:** full training + probe suite runs only on Devanagari
(primary) and Bengali (cross-script structural check). Santhali (Ol
Chiki) and Kashmiri (Perso-Arabic-derived) get a lightweight zero-shot
test only — render, run inference, check calibration — with no training
pipeline built for them.

**Alternatives considered:** (a) Devanagari + Bengali only, no coverage
of the extreme languages at all; (b) full training pipelines for 4+
scripts.

**Why (a) was rejected:** Sarvam's own published spread — the thing this
whole project exists to explain — has Kashmiri at 55.93 and Santhali at
80.32 as the actual outliers. A project that never touches those two
scripts can gesture at the motivation but can't speak to it directly in
the room.

**Why (b) was rejected:** compute and font-availability cost scales
faster than the marginal insight. The zero-shot floor test answers the
specific question that matters here — does confidence collapse correctly
at true zero exposure — without needing a trained model per script.
Devanagari alone provides the statistical power for Probe 1 (hundreds of
conjunct classes spanning orders of magnitude in natural frequency), so
the causal experiment doesn't need more scripts to be well-powered.

**Credit:** this correction came from a direct question about scope,
2026-08-24.

**Verified:** 2026-08-25. Numbers re-confirmed directly against Sarvam
Vision's own launch blog (sarvam.ai/blogs/sarvam-vision, Feb 5 2026);
Kashmiri 55.93 / Santhali 80.32 exact match; spread holds through
Sarvam's own newest frontier model.

---

### 7. CER/error alignment at grapheme-cluster level, not code-point level

**Decision:** all error taxonomy and metric computation (Stage 0
onwards) aligns predictions to ground truth at the grapheme-cluster
level.

**Why:** a single visual character (akshara) in Devanagari can span
multiple Unicode code points. Code-point-level CER double-counts or
mis-attributes errors within one visual unit. A 2026 Devanagari OCR
benchmark paper flags exactly this as an open limitation in comparable
work — grapheme-aware scoring, not code-point CER, is the correct choice
and is explicitly named there as unfinished work in the literature.

---

### 8. Tier 2 phonetic equivalence via transliteration, target scheme ISO 15919

**Decision:** Tier 2 (phonetic/spelling equivalence, as opposed to
Tier 1's deterministic encoding equivalence) is implemented by
transliterating both reference and hypothesis into a shared Latin
canonical form and comparing there, using ISO 15919 as the target
scheme.

**Alternatives considered:** (a) a hand-built rule table enumerating
every phonetic-equivalence case, the way Tier 1 handles encoding
equivalence; (b) a lossy/simplified romanization scheme (Hunterian,
ITRANS-style) as the transliteration target; (c) using an LLM as the
primary scorer.

**Why not (a):** phonetic equivalence classes are open-ended (alternate
valid spellings of names, transliteration ambiguity) in a way encoding
equivalence isn't — a rule table would never be complete. A general
transliteration-then-compare mechanism handles the open-ended case in
one pass, and generalizes to additional Brahmic scripts for free (no new
rule table needed per script).

**Why not (b):** simplified romanization schemes collapse real
distinctions (short/long vowels, retroflex vs. dental) that should count
as genuine errors, not equivalences. ISO 15919 is close to lossless for
Indic-to-Latin round-tripping, which is the property this needs.

**Why not (c) as primary:** LLM judgments aren't deterministic or free,
and a number that can't be regenerated identically from a script is a
weak thing to defend in an interview. LLM-as-judge is used instead as a
validation pass over the cases where the deterministic method disagrees
with a small hand-labeled set — the same pattern used in a 2026 Indian
financial-text benchmark, where automated scoring was the headline
metric and LLM-as-judge only audited the disagreements.

**Boundary:** this mechanism only applies within the Brahmic script
family (Devanagari, Bengali, and related). It doesn't extend to Ol Chiki
or Perso-Arabic Kashmiri — no shared transliteration target exists — so
Probe 5b (zero-shot floor) doesn't use Tier 2 scoring at all; it's a
confidence-calibration question, not a text-matching one.

**Credit:** this design came directly from a suggestion to use
transliteration-based canonical equivalence rather than an enumerated
Unicode rule table, 2026-08-24.

---

### 9. Renderer layouts sourced from real documents, not invented templates

**Decision:** `layout_sources.py` pulls actual document layouts —
Internet Archive scanned Indic books, India.gov.in multilingual PDFs,
Wikipedia Indic-language articles — rather than hand-designing template
categories.

**Why:** synthetic renders that don't reflect real document structure
overstate every system's quality, per direct benchmark evidence (a 2026
Devanagari paper found ten systems cluster tightly on clean synthetic
text but spread across a 76-point range on real scans). Real layouts as
the _source_, controlled-only where a variable genuinely needs
isolating (glyph frequency for Probe 1), is the right split.

---

### 10. The naturalness confound in glyph-frequency resampling

**Decision:** `glyph_frequency.py` resamples source text toward a target
distribution while trying to preserve local language-model plausibility
as much as the target mode allows, and a blank/noise-image control
(Probe 3) is run alongside every condition specifically to separate
"reads the glyph" from "guesses from language prior."

**Why this matters:** reshaping glyph frequency necessarily changes how
linguistically natural the resulting text is, which confounds with the
exposure manipulation — a flattened-frequency page might also just be a
harder page for the decoder's language model, independent of vision.
Probe 3 exists specifically to give a per-condition baseline for "how
much of this model's apparent reading is actually guessing," so the
confound can be measured rather than ignored.

---

### 11. RLVR ablation scope: coverage term only

**Decision:** exactly one RLVR ablation — remove the coverage penalty,
show and quantify the model learning to omit text to game accuracy.

**Alternatives considered:** a full sweep across every reward component
(character accuracy, TEDS, rank correlation, coverage) individually.

**Why:** full-sweep ablations are expensive relative to what they'd add
here. The coverage-term removal is cheap (a single retrain), produces a
specific and checkable failure mode, and is a better interview story
than a matrix of marginal deltas — a mechanism, not a table.

---

### 12. Table-to-prose scoped down to header-cell binding accuracy

**Decision:** the original table-to-prose idea (generate a sentence per
row) is cut down to a single checkable metric: after OCR, is each cell
still correctly bound to its column header?

**Why:** prose generation needs its own evaluation methodology (what
counts as a correct paragraph?) that this project doesn't have budget
for. Binding accuracy has clean ground truth from the renderer, is a
direct measure of whether table _structure_ survived OCR — which was
the actual motivation (tables get destroyed by naive RAG chunking) — and
gives something concrete to compare against Sarvam's Extract endpoint,
which already returns structured fields with per-field confidence.

---

### 13. Comparing against Sarvam's Extract endpoint for structured output

**Decision:** Stage 3's table-binding metric and Stage 5's transfer
analysis compare against Sarvam's Extract (schema-based key-value
extraction with per-field confidence), not just Digitise (full-document
OCR), where structure is the relevant question.

**Why:** Extract is the closer real-system analogue to "did structure
survive," and its exposed per-field confidence is directly comparable
to the instrument's own calibration output (Probe 5).

---

### 14. Three seeds per Probe 1 condition, non-negotiable

**Decision:** each of the three glyph-frequency conditions
(natural/flattened/inverted) trains with 3 random seeds, 9 runs total,
not 3.

**Why:** with a single seed per condition, the entire observed spread
could be seed noise rather than signal. The instrument is small enough
(30-60M params, T4-trainable in under an hour) that this is affordable,
so there's no excuse to skip it — it's the difference between a finding
and a coin flip.

---

### 15. Sarvam transfer comparison runs on Tier B (degraded), not just Tier A (clean)

**Decision:** Stage 5's transfer analysis runs on both clean (Tier A)
and degraded (Tier B) renders, not clean alone.

**Why:** direct benchmark evidence shows systems that look nearly
identical on clean synthetic text spread dramatically once degradation
or real scans enter the picture. A clean-only comparison risks
concluding "no difference" purely because the test was too easy to
separate anything.

---

### 16. Cascade demo (Stage 6) reports router quality, not cost savings

**Decision:** the escalation-cascade result is framed as "does the
instrument's confidence correctly identify which pages need Sarvam,"
compared against three baselines (random, layout-complexity,
Tesseract-confidence), rather than as a headline cost-reduction number.

**Why:** the instrument is expected to be considerably less accurate
than production OCR engines (it's built to be probed, not to compete —
see `README.md`). A cost-savings framing built on top of a weak base
model falls apart under one follow-up question. A router-quality framing
survives regardless of the instrument's absolute accuracy, because it's
measuring whether the _confidence signal_ is meaningful, which is the
actual question Sarvam's own backend team's harness thesis depends on.

---

### 18. Tier 2 scope clarified: strict phonetic identity, not conventional spelling variance

**Decision:** Tier 2's validation set must only contain pairs that are
truly phonetically identical (same sounds), not pairs that are merely
conventionally used interchangeably despite differing sounds.

**Why:** the first validation set (4 of 8 pairs) mixed these up —
short/long vowel pairs and an added-syllable spelling were included as
"equivalent," and ISO 15919 correctly refused to collapse them, since
those pairs really do differ phonetically. This isn't a tool bug. Real
examples of strict phonetic identity in Devanagari: anusvara vs. the
homorganic nasal consonant, e.g. हिन्दी vs. हिंदी (both "Hindi",
genuinely identical pronunciation) — these belong in the validation
set. Genuinely different-sounding pairs, however conventional, do not.

---

### 19. Every Sarvam page fetched exactly once, cached, all sweeps run offline

**Decision:** `sarvam_client.py` never re-fetches a page. Every
threshold sweep, every comparison, every re-analysis in Stage 6 runs
against `data/cache/`, never against a fresh API call.

**Why:** the entire project's paid budget is ~200 pages. An escalation
sweep tested naively (one API call per threshold value) would blow
through the budget on a single experiment. Caching turns N-pages-per-
threshold into N-pages-total.

---

### 20. Hand-review suggestions live in `hand_review_assist.py`, not inside the viewer loop

**Decision:** keep `hand_review.py` as the interactive viewer and put
the UNEXPLAINED-case label heuristic in a new file,
`src/eval/hand_review_assist.py`. The viewer imports it and prompts
per unexplained engine output. Enter confirms a real suggestion; `s`
skips; a typed string overrides. Notes rows include
`suggestion_outcome` ∈ {`agent-suggested-and-confirmed`,
`human-overridden`, `skipped`}.

**Alternatives considered:** (a) fold the heuristic into
`hand_review.py` next to the input() loop; (b) auto-write the suggested
label with no keypress; (c) invent extra buckets when the four don't
fit; (d) call an LLM for the suggestion.

**Why not (a):** the heuristic has constructed pairs with a right
answer and needs a validation entry point that doesn't start a review
session. Mixing that into the prompt loop would also make it easier to
"just extend the category list" the next time a pair is awkward.

**Why not (b):** the suggestion is a proposal. Auto-accept would make
the notes file an agent artifact pretending to be a hand taxonomy, and
we would have no later check of how often the guess was right.

**Why not (c):** IMPLEMENTATION.md Stage 0's residual buckets are the
fixed set. Forcing a match (or growing the set ad hoc) hides the cases
the taxonomy doesn't cover yet.

**Why not (d):** non-deterministic, not free, and a label we can't
regenerate identically is a weak thing to defend — same reason Tier 2
doesn't use an LLM as the primary scorer (decision #8).

**Heuristic priority** (first match): repeated/looped text, then same
token-multiset in a different order, then bases identical after
NFD-stripping matras and nuktas only (virama/anusvara/visarga are
*not* stripped — those are different errors), then similar-length
letter substitution, else explicit no-fit.

**Prompt change:** Enter used to skip. It now confirms when a
suggestion exists; skip is `s` / `skip`. Empty prediction vs non-empty
ground truth is no-fit (omission isn't one of the four).

---

### 21. Layout extraction: PDF text-layer for born-digital, ink projection for scans

**Decision:** `layout_sources.py` uses two extractors that share one
classifier. Born-digital pages (Wikipedia print PDFs, government PDFs)
are read with pymupdf's text/table/widget APIs. Scanned IA pages have
no usable text layer, so columns and margins are inferred from 1-D ink
projection profiles. Wikipedia is fetched as REST printable PDFs (not
HTML screenshots). IA is fetched as IIIF page images at ~800px, not
full DLI PDFs. The india.gov.in source is operationalized as a public
`.gov.in` PDF (NCERT Hindi textbook) because india.gov.in is a portal
and the layout we need lives on the linked government file. Templates
are cached in `data/cache/layouts/`; HTTP is cache-first.

**Alternatives considered:** (a) hand-drawn region templates labelled
single-column / two-column / etc.; (b) a learned layout detector
(DiT, LayoutLMv3); (c) always rasterize and only use projections;
(d) download full IA PDFs.

**Why not (a):** DECISIONS.md #9 — invented geometry overstates every
system's quality. **Why not (b):** Stage 1 has to be deterministic and
inspectable; a second neural net's errors would leak into Probe 1.
**Why not (c):** digital PDFs already know their block boxes; throwing
that away adds error. **Why not (d):** DLI PDFs are often 100MB+ for
near-duplicate pages; IIIF gives the few pages the bank actually needs.

**Classifier detail:** category priority is form → table-embedded →
two-column → marginalia → single-column. Tiny Wikipedia infobox tables
do not count as `table-embedded` (need height ≥ 0.18 and width ≥ 0.25,
or combined table area ≥ 0.08) — otherwise every wiki article would
be labelled a table page and the two-column reading-order problem
would disappear from the bank.

**Verified:** 2026-08-24, unit tests on synthetic two-column/form PDFs
plus a live fetch of `hi.wikipedia.org` "भारत" PDF and two IA page
images from `in.ernet.dli.2015.480257`.

---

### 22. hand_review_assist must apply Tier 0 whitespace normalization before diffing

**Decision:** `suggest_unexplained_label()` in `hand_review_assist.py`
calls `normalize_whitespace()` from `equivalence_tables.py` on both
strings before any heuristic comparison — the same Tier 0 pass that
`classify_against_tiers()` in `hand_review.py` applies before Tier 1
and Tier 2.

**Bug found (2026-08-24):** the assist module only used `.strip()`.
Internal newlines or extra spaces from line-wrapping were therefore
scored as letter-level substitutions (`genuine-misread`) even when
the character content matched.

**Measured on first 80 Tesseract Hindi rows**, reproducing the pre-Tier-0
UNEXPLAINED pool (65 cases — the count before
`normalize_whitespace()` existed anywhere in the pipeline):

| Heuristic | genuine-misread | dropped-matra-nukta | no-fit |
|-----------|-----------------|---------------------|--------|
| Before (`.strip()` only) | 63 | 2 | 0 |
| After (`normalize_whitespace()`) | 52 | 3 | 10 |

The 10 `no-fit` cases are whitespace-only diffs that now correctly
return “should have been EXACT MATCH” instead of being forced into
`genuine-misread`. **Whitespace noise was inflating the misread count
by ~17% of the UNEXPLAINED pool** (11 of 65 cases moved out of
genuine-misread: 10 to no-fit, 1 to dropped-matra-nukta).

With Tier 0 also wired into `classify_against_tiers()` and
`normalize_tier1()`, those 10 whitespace-only pairs no longer reach
UNEXPLAINED at all (55 UNEXPLAINED remain; suggestion distribution
52 / 3 on that smaller pool).

**Validation:** added constructed pair `("राम सीता", "राम\\n\\nसीता")`
→ no-fit; `hand_review_assist.py` validation now 12/12.

---

### 23. genuine-misread heuristic false positives are Tier 1 punctuation gaps, not order/matra mislabels

**Finding (2026-08-24 hand-review pass):** after the Tier 0
whitespace fix (#22), a fresh Hindi+Bengali review of all 130
UNEXPLAINED cases (82 Hindi, 48 Bengali, three engines) found
**zero** cases where `genuine-misread` should have been
`reading-order-break` or `dropped-matra-nukta`. The assist module's
priority order for those two buckets is working.

**Second bug, different shape:** ~13 cases (7 Hindi, 6 Bengali) where
the only residual difference is punctuation spacing or symbol choice
(` ।` vs `।`, `|` vs danda, trailing sentence mark omitted). Tier 1
normalizes danda→period but not pipe→period or space-before-danda
inconsistency, so these pairs stay UNEXPLAINED; low edit distance
then fires `genuine-misread`. Examples: Hindi ids 222–225 (masjid
sentences — letter content identical, danda spacing differs).

**Hand-review action:** those cases were **skipped** in
`hand_review_notes.jsonl` (`suggestion_outcome: skipped`), not
confirmed as genuine-misread. Fix belongs in `equivalence_tables.py`
(Tier 1 expansion: pipe as danda alias, collapse space before danda),
not in lowering the assist heuristic's edit-distance threshold.

**Confirmed genuine-misread rate:** 70/77 Hindi suggestions, 36/42
Bengali — the bulk of suggestions are real letter-level OCR noise
(e.g. Tesseract id=138 plain/degraded: whole-clause garbling).

---

### 24. Degradation parameters inverted from measurements, not hardcoded; calibration is a born-digital full page

**Decision:** `degradation_profile.py` stores an *empirical joint*
distribution (bootstrap over measured pages), not four independent
constants. Apply-units are PIL Gaussian radius, additive Gaussian
std on 0–255, CCW skew in degrees, show-through blend alpha.
Blur sigma is inverted from Laplacian variance against a calibration
curve built by blurring a Wikipedia-print PDF raster at ~120 dpi —
the same scale as the 1600px-wide IA measurement scans. Line-level
GlotOCR `*_plain.png` images are the wrong scale and were tried
first; they reported sigma≈0 on every real book page.

IA pages are fetched at 1600px into `data/cache/degradation/scans/`,
separate from the 800px layout thumbnails. Downsampling to layout
size is itself a blur and would contaminate the measurement.

Well-scanned DLI JPEGs have near-zero paper noise and show-through
(the verso ghost does not survive IIIF JPEG). The profile therefore
always mixes in a capped set of GlotOCR `img_old_document` pages as
the heavier tail. This is not a substitute for prescriptions /
photocopied forms — those can be dropped into `scans/` later without
code changes. Joint bootstrap (sample a whole page's four-tuple)
keeps blur and noise correlated, unlike independent marginal draws.

**Verified:** 2026-08-24. Unit tests recover known sigma/skew/noise
on synthetic pages. Fitted profile n=22 (10 IA 1600px + 12 GlotOCR
degraded): blur median 1.16, noise median 1.35, skew p90 ≈ 5.6°,
show-through median 0.044.

---

### 25. Glyph-frequency resampling is sentence-level importance sampling over Indic clusters only

**Decision:** `glyph_frequency.py` counts and targets Unicode grapheme
clusters (`regex` `\X`) that contain at least one Indic-script code
point (Devanagari, Bengali, Ol Chiki, Arabic, …). Whitespace and Latin
punctuation / digits are kept in the text but excluded from the dial —
otherwise inverted mode promotes `%` instead of rare conjuncts.
Resampling is sentence-level importance sampling (arithmetic-mean
target/natural ratios, sharpened by a mode-dependent power) so local
LM structure survives (DECISIONS.md #10). `inverted` adds a residual
bigram-constrained repair pass; `flattened` does not (repair ate
entropy on the toy corpus). Flattened target is 65% uniform / 35%
natural, not pure uniform — pure uniform is unreachable without
inventing text.

**Alternatives considered:** (a) cluster-level random substitution as
the primary mechanism; (b) geometric-mean sentence weights; (c)
counting every non-whitespace grapheme including Latin.

**Why not (a):** destroys local plausibility, which Probe 3 then
cannot separate from the exposure effect. **Why not (b):** on real
GlotOCR sentences a single rare cluster cannot overcome ten common
bases in a product. **Why not (c):** observed on the first Hindi smoke
run — inverted chased `%` / digits.

**Tolerance honesty:** on the 60-line Hindi GlotOCR slice, flattened /
inverted TV to target stays high (~0.3–0.7) because every sentence is
already a natural mix; the dial *direction* is verified (uniform TV
drops; rare-quartile mass rises) and the toy corpus hits the
acceptance checks. Probe 1 will feed a larger rendered corpus.

---

### 26. Tier 1 gap closed: pipe-as-danda and space-before-punctuation

**Decision:** added pipe/double-pipe as danda-equivalent, and a regex
pass stripping whitespace before terminal punctuation, both applied
after Tier 1's other pair substitutions.

**Why:** Stage 0 hand-review found 13 of 130 UNEXPLAINED cases were
neither Tier 1 nor Tier 2 despite being pure formatting noise --
engines writing "|" for danda, or a stray space before it. Confirmed
against the exact reported failing case (id=222) before shipping.

**Implements:** the Tier 1 expansion recommended in #23. Pipe maps to
`.` (same canonical as danda); `||` maps to `..`. The regex pass runs
after pair substitution so `है ।` and `है।` both normalize to `है.`.

**Verified:** 2026-08-24. `equivalence_tables.py` smoke tests include
id=222 and id=223 pairs; `classify_against_tiers()` no longer marks
those Tesseract Hindi rows UNEXPLAINED.

---

### 27. Renderer: uharfbuzz for metrics/boxes, Pillow+raqm for paint; Tier A is clean

**Decision:** `render.py` shapes with uharfbuzz (glyph advances +
cluster IDs → per-grapheme-cluster bounding boxes) and paints with
Pillow's FreeType/raqm path (correct conjunct drawing without
hand-rolled matra placement). Fonts are discovered from an ordered
candidate list (Kohinoor / Sangam on macOS, Noto on Linux/Colab).
Tier A defaults to zero degradation (fixed and clean) so Probe 1's
only moving part is glyph-frequency mode; Tier B samples the measured
profile; Tier C is passthrough of real images with empty cluster
boxes (Probe 6 needs text, not boxes).

**Alternatives considered:** (a) Pillow alone for both paint and boxes;
(b) applying the profile median as Tier A's fixed damage; (c)
freetype-py glyph bitmaps.

**Why not (a):** Pillow does not expose per-cluster advances cleanly.
**Why not (b):** a non-zero fixed blur is still a confounder when we
can just hold damage at zero. **Why not (c):** more moving parts; HB +
Pillow already covers Indic.

**Verified:** 2026-08-24. Unit tests: page in ≪1s, conjunct `क्ष` gets
a box, Tier B differs from clean, natural vs inverted changes the
realized cluster bag.

---

### 28. run_baselines engine APIs: surya.recognition + PaddleOCR 3.x predict()

**Decision:** fix two package-version regressions in `run_baselines.py`
without pinning older packages.

1. **Surya (surya-ocr 0.22.1):** import was a typo —
   `surya.rognition` → `surya.recognition`. The 0.22 API is
   full-page `RecognitionPredictor(SuryaInferenceManager())` with
   `full_page=True`; DetectionPredictor + `text_lines` is gone.
   Output is HTML per layout block — flatten with BeautifulSoup.
   Requires `llama-server` on PATH (`brew install llama.cpp`).

2. **PaddleOCR (3.7.0):** drop `show_log` (removed); drop
   `use_angle_cls` / `ocr(..., cls=True)` in favour of constructing
   with `lang` only (and disabling doc preprocess for speed);
   call `predict()` instead of deprecated `ocr()`. Hindi lang code
   is `hi`, not the old `devanagari` alias (which now raises
   "No models are available").

Predictors are process-level singletons so a 10-image smoke does not
reload weights ten times. CLI gains `--engine` / `--language` /
`--limit` / `--variant` for subset runs.

**Verified:** 2026-08-24. Hindi plain, 10 images each —
`paddleocr` 10/10, `surya` 10/10, zero `skipped_reason`. Full corpus
still needs a fresh overwrite of `data/predictions/{surya,paddleocr}/`
(the smoke truncated those jsonl files to 10 rows).

---

### 29. Glyph-frequency control via bigram-guided synthesis, not sentence IS alone

**Decision:** `flattened` / `inverted` allocate an exact integer glyph
multiset from the target PMF (largest-remainder), then pack those
glyphs into sentence-shaped strings with a bigram walker trained on
the source corpus. `flattened` target is pure uniform over the
observed Indic support. Explicit acceptance: TV(realized, target) ≤
0.08 (`TARGET_TV_TOLERANCE`).

**Investigation (before the change, Hindi n=60):**

1. Flattened TV≈0.36 was a **corpus-size / composition limit**, not
   a tuning miss. A greedy oracle that only selects existing sentences
   floors at TV≈0.29 (flat) / ≈0.73 (inverted). Max rare-quartile
   fraction in any single sentence is 0.33; median 0.0. Importance
   sampling cannot leave the convex hull of sentence bags.
2. The old inverted `repair` pass **helped** frequency control (TV
   dropped ~0.10–0.13 across seeds). The audit's 40/60 mutated
   sentences were working toward the target, not diluting it. The
   residual gap after repair was still ~0.65–0.70 because IS started
   too far from the target.

**Alternatives considered:** (a) sharper IS weights / more repair
swaps; (b) synthesis as adopted; (c) keep 65/35 flat mix from #25.

**Why not (a):** cannot beat the oracle floor. **Why (b):** Probe 1
needs the histogram to *be* the experimental variable. **Why not (c):**
with synthesis, pure uniform is reachable and is the honest
"flattened" condition.

**Naturalness tradeoff:** packing is only as LM-plausible as the
bigram table allows; this is the #10 confound, measured later by
Probe 3 — not ignored.

**Before → after (Hindi n=60, seed 0):**

| mode | TV before | TV after | rare-q mass before | after |
|------|-----------|----------|--------------------|-------|
| natural | 0.000 | 0.000 | 0.053 | 0.053 |
| flattened | 0.361 | 0.050 | 0.048 | 0.246 |
| inverted | 0.657 | 0.005 | 0.160 | 0.428 |

---

### 30. Stage 1 IMPLEMENTATION checkboxes corrected to match 2026-08-24 audit

**Decision:** flip Stage 1 status markers to match the audit, not the
earlier overclaim. `layout_sources.py` and `degradation_profile.py`
→ `[~]` PARTIAL (real code paths, incomplete source/bank coverage).
`glyph_frequency.py` → `[x]` after #29. `render.py` and tiers A/B/C
→ `[x]` (verified DONE). Tolerance for the glyph histogram is now
stated explicitly in IMPLEMENTATION.md acceptance (TV ≤ 0.08).

**Why:** AGENTS.md requires `[x]` only when verified. Leaving PARTIAL
items checked would let Probe 1 / Stage 5 proceed on a false
"layouts and degradation are done" reading.

---

### 31. run_baselines resumes by append+skip, not overwrite

**Decision:** `run_baselines.py` opens each
`data/predictions/{engine}/{language}.jsonl` in append mode, skips any
`(engine, language, id, variant)` already present, prints per-image
progress, and flushes/fsyncs after every line. Incomplete trailing
lines from a mid-write kill are truncated on the next startup.
`OCR_PRED_ROOT` overrides the output root for isolated smoke tests.

**Alternatives considered:** (a) keep overwrite-on-rerun and add an
explicit `--resume` flag; (b) write a sidecar `.done` checkpoint file;
(c) buffer all results then write once.

**Why:** AGENTS.md "Long-running scripts" makes resume-by-default the
required behaviour — re-running the same command is the recovery path.
(a) makes the unsafe path the default; (b) duplicates state already in
the jsonl; (c) loses the whole batch on Ctrl-C. To redo from scratch,
delete the jsonl.

---

### 32. Heavy scripts: one `--data-root`, Colab export into IMPLEMENTATION.md paths

**Decision:** CPU/GPU-heavy scripts (OCR batches, training) are written
for Colab, not assumed to run on the laptop. Inputs and outputs hang
off one `--data-root` / `OCR_DATA_ROOT` (default `data/`). The last
step zips or copies results so they unpack at the path already named
in IMPLEMENTATION.md. `run_baselines.py` already used `OCR_PRED_ROOT`
as a single output root; Colab now uses `--data-root` for both
`{root}/raw` and `{root}/predictions`, with `--pred-root` /
`OCR_PRED_ROOT` kept as a nested smoke-test override. Jsonl
`image_path` is the repo-canonical `data/raw/...` path, not a Colab
absolute path. `--export-zip` writes archive members
`data/predictions/{engine}/{language}.jsonl`.

**Alternatives considered:** (a) separate `--raw-root` and
`--pred-root` as the Colab knobs; (b) write Colab-absolute paths into
the jsonl; (c) Drive-only, no zip.

**Why:** (a) is two paths to change between machines, which is what
the "one configurable root" rule is there to avoid; `--pred-root`
stays as an *extra* override so a resume smoke test never writes into
a live tree. (b) breaks `hand_review.py` after download. (c) is the
other allowed export, but `files.download()` needs a local file.

---


### 33. Per-image OCR hard timeouts, resume-safe

**Decision:** wrap each individual image invocation inside
`run_baselines.py` with a hard per-image timeout, recording
`skipped_reason="timeout"` on expiry and continuing to the next
image.

The timeout is implemented via `signal.setitimer()` (with a
`SIGALRM` handler) so one stuck/out-of-distribution image cannot
block a multi-hour OCR batch indefinitely.

**Alternatives considered:** (a) multiprocessing and killing a worker
process; (b) external watchdog + subprocess; (c) only logging slow
progress / best-effort heuristics.

**Why:** (c) doesn't satisfy the "no single image can ever hang a
multi-hour run again" requirement. (a)/(b) can be more robust but add
operational complexity and/or overhead; for the first fix we use a
simple signal-based hard stop that should interrupt the Python call
path. If we later find signals don't reliably cut off a specific
backend hang, we can escalate to process killing for that engine.


---

### 34. Per-image timeouts: OS-process isolation + forced kill

**Decision:** replace the SIGALRM-based timeout in `run_baselines.py` with
per-image OS-process isolation:

1. Run `engine_fn(image_path, language)` in a `multiprocessing.Process`.
2. Enforce the timeout with `process.join(timeout=N)`.
3. If still alive, call `process.terminate()` and then `process.kill()`
   if it does not exit within a short grace window.
4. Send the result back via a `multiprocessing.Queue`; if nothing is sent
   before the timeout window closes, record `skipped_reason="timeout"`.

**Why:** Python signals do not reliably interrupt some real backend hang
patterns (notably when the stuck path is inside a threadpool shutdown /
thread.join() / lock acquisition chain). Killing the whole OS process
reliably terminates the blocked thread no matter what it is waiting on.

**Alternatives considered:** (a) keep SIGALRM + add more threading cancellation
logic (does not fix the root signal-interruption limitation); (b) watchdog
subprocess without process-level kill (still leaves the blocked thread);
(c) make the engine itself cooperative/cancelable (not feasible for closed
backends or third-party OCR libraries).

---

### 35. Taxonomy recomputes Tier 1/2 live; notes only label residuals

**Decision:** `error_taxonomy.py` re-runs current Tier 1 and Tier 2
code on every prediction. `hand_review_notes.jsonl` is used only for
rows that remain unexplained, as the genuine-error label (or
UNREVIEWED if no note exists).

**Alternatives considered:** (a) trust the bucket stored in the notes
file at review time; (b) drop rows with no note instead of UNREVIEWED.

**Why:** notes were captured while Tier 1/2 were still being patched
(#18, #22, #23, #26). Re-scoring means a table fix updates the
headline fractions without another human pass. Silent drop would
inflate “exact” and hide coverage gaps.

---

### 36. Instrument encoder 320-d / decoder 384-d, bridged in `train.py`

**Decision:** ViT encoder `d_model=320`, decoder `d_model=384`, with a
linear projection on encoder memory inside `InstrumentModel` in
`train.py` (not inside encoder.py or decoder.py).

**Alternatives considered:** (a) one shared width; (b) put the
projection in the decoder constructor.

**Why:** decoder does more work at this size (see module docstrings).
Keeping the bridge in the combined model leaves encoder.py/decoder.py
independently smoke-testable.

---

### 37. Instrument training consumes line-crop manifests, not page renders

**Decision:** `train.py` reads JSONL `{"image_path","text"}` line crops.
Stage 1’s `render.py` still emits page images plus per-cluster boxes.
No adapter is in-repo yet; real training is blocked on that handoff.

**Alternatives considered:** (a) train on full pages immediately;
(b) silently assume Stage 1 already writes line crops.

**Why:** IMPLEMENTATION.md says line-level first. Flagging the
mismatch in the train.py module docstring (and here) is better than
pretending `make probe1-smoke` is a real Probe 1.

---

### 38. `generate.py` is greedy, no KV cache, returns probe-facing tensors

**Decision:** add `generate.py` (not in the original spec): greedy
decode only; recompute the full decoder forward each step; return
text, ids, per-step confidence, and top-k.

**Alternatives considered:** (a) beam search; (b) KV cache; (c) fold
generation into `train.py`.

**Why:** Probes 2/3/5 need distributions and confidence, not prettier
strings. Greedy is deterministic. KV cache is a real speed win later;
flagged as skipped at this scale rather than silently omitted.

---

### 39. Probe 1 orchestrator is in-process; checkpoint step is “done”

**Decision:** `probe1_exposure.py` calls `train()` in-process for all
9 (condition, seed) jobs. A run is complete iff the checkpoint exists
and `step >= total_steps`. No sidecar done-file.

**Alternatives considered:** (a) nine subprocesses; (b) a separate
`done` marker.

**Why:** subprocesses re-pay torch import nine times on a short Colab
clock. Two sources of truth (checkpoint vs marker) drift. This layer
is on top of train.py’s mid-run resume, not a replacement.

---

### 40. `make smoke-test` is the no-compute architecture proof

**Decision:** a `Makefile` with `smoke-test` (tier self-tests +
instrument `__main__` + fake Probe 1 + generate) and separate
`stage0-*` real-data targets. Fake crops live in
`scripts/make_fake_probe1_data.py`.

**Alternatives considered:** (a) document Colab-only until real data
exists; (b) check in a tiny real subset.

**Why:** the interview needs a command that proves the pipeline is
wired without GPU or `data/raw`. Fake data is explicitly not a
finding. Known gaps: `probe1-smoke` looks for the orchestrator under
`src/models/instrument/`; the file is `src/probes/probe1_exposure.py`.
The fake-data script on disk is currently corrupt (TODO.md).

---

### 41. Line GT boxes from `measure_line_width`, not a second shape pass

**Decision:** inside `render_page`'s paint loop, each wrapped line gets a
`LineGT` with width from the existing `measure_line_width()` helper and
a vertical pad of `1.05 * font_size` above / `0.35 * font_size` below
the baseline. `PageGT.lines` is filled for Tier A/B; Tier C stays empty
(real scans have no synthetic line boxes).

**Alternatives considered:** (a) derive line boxes by unioning cluster
bboxes on that line; (b) call HarfBuzz again with a separate buffer;
(c) use Pillow `textbbox` only.

**Why:** (a) undersizes ascenders/matras when cluster clips are tight;
(b) duplicates work already paid by wrap + cluster shaping; (c) can
disagree with HarfBuzz advances on Indic. Reusing `measure_line_width`
keeps paint metrics and GT width on the same font. This is the page-side
half of the Stage 2a line-crop contract (#37); cropping pages into
manifests is still a separate adapter.

---

### 42. PaddleOCR 3.7: no legacy kwargs; disable MKLDNN on CPU

**Decision:** `_get_paddle_ocr` constructs PaddleOCR 3.7 with only
supported kwargs: `lang`, `use_doc_orientation_classify=False`,
`use_doc_unwarping=False`, `use_textline_orientation=False`,
`enable_mkldnn=False`. Call site stays `predict()`. Never pass
`show_log` or `use_angle_cls` — both raise
`ValueError("Unknown argument: ...")` via
`paddleocr._common_args.parse_common_args`.

**Alternatives considered:** (a) pin paddleocr 2.x so the old kwargs
work; (b) leave MKLDNN on and accept intermittent CPU failures;
(c) catch Unknown-argument and retry without kwargs.

**Why:** Colab/local batches that still used `show_log` wrote an entire
`data/predictions/paddleocr/*.jsonl` of nulls. After dropping
`show_log`, a Hindi resume then hit OneDNN
`ConvertPirAttribute2RuntimeAttribute` on CPU wheels; `enable_mkldnn=False`
keeps the same jsonl schema and `predict()` protocol while avoiding that
backend crash. GPU Colab ignores MKLDNN. Not a silent protocol change —
engine kwargs only.

**Verified:** 2026-08-24. Isolated smoke (`--pred-root`, `--limit 2`) —
Hindi 2/2 and Bengali 2/2 non-null `predicted_text`, schema unchanged;
synthetic engine failure still records `skipped_reason`. Live
tesseract/surya/paddleocr prediction files left untouched.

---

### 43. Scaled line-manifest export uses `--data-root`, not repo `--root`

**Decision:** `export_manifest_scaled.py` takes `--data-root` /
`$OCR_DATA_ROOT` (default `data/`), same as `run_baselines.py`
(DECISIONS.md #32). Layout: `{root}/raw/.../ground_truth.jsonl`,
`{root}/cache/line_crops/{script}/`, `{root}/manifests/{script}_{mode}.jsonl`
plus `{script}_{mode}.progress.json`. Imports follow probe1's pattern:
insert `src/` on `sys.path` from `__file__`. Page RNGs use sha256 of
`(script, mode, page_idx)`, not Python's salted `hash()`.

**Alternatives considered:** (a) `--root` = repo root with hard-coded
`data/` underneath (draft CLI); (b) invent a third path flag.

**Why:** (a) disagrees with AGENTS.md's one-knob Colab rule and with
`export_line_manifest.py`'s documented `--data-root`-style tree.
Stable seeds keep a resumed page reproducible if it is ever re-rendered.

**Verified:** 2026-08-25. Hindi `--pages-per-mode 3`: natural 76 /
flattened 83 / inverted 87 lines; interrupt mid-natural then resume
appended only remaining pages (0 path dupes); full rerun printed
`already complete — skipping` for all three modes with unchanged counts.

---

### 44. Canonical line-crop height 70 px (5 × PATCH_SIZE)

**Decision:** `export_line_crops()` pads every saved line PNG to
**70 px** tall (white background, top-aligned ink) before writing to
`data/cache/line_crops/`. Width stays variable; width padding remains
in `collate_batch`.

**Alternatives considered:** (a) 64 px from train.py's collate docstring
("e.g. 64px, per docs/stage2_design_notes.md"); (b) leave natural
43–44 px and fix collate_batch to tolerate mixed heights; (c) round
observed max up to 56 (4 × 14).

**Why:** `docs/stage2_design_notes.md` is not in the repo — 64 was an
example, not a settled spec. Dry-run crops were 43–44 px (222/266 at
43, 44 at 44), which breaks `collate_batch`'s assumption that
`height = images[0].height` for the whole batch and makes patch-count
masks wrong when heights differ. 70 = smallest multiple of
`PATCH_SIZE=14` above the observed range; it matches existing smoke
fixtures (`encoder.py` `height=70`, `generate.py` blank tensor,
`make_fake_probe1_data.py`). `PatchEmbedding` uses
`Conv2d(..., stride=patch_size)` — partial rows are silently dropped,
not errored, so fixed height must be enforced at export time.

**Verified:** 2026-08-25. Hindi dry run `--pages-per-mode 3`: all 266
crops height 70 px (was 43/44 mixed); widths still 90–879.

---

### 45. Probe 6: resize Tier C to 70 px; score synthetic on held-out pages only

**Decision:** Before instrument evaluation on Tier C real images, resize
to the same **70 px** canonical height used for synthetic line crops
(`resize_to_canonical_height` in `probe_utils.py`, aspect ratio
preserved). For the instrument's *synthetic* side of Probe 6, do **not**
score against the training manifests (pages 0–99); render held-out pages
**100–109** via `export_manifest_scaled.py --pages-per-mode 110` and
evaluate only those unseen lines.

**Alternatives considered:** (a) feed raw Tier C heights (e.g. 254 px)
straight into the encoder; (b) measure synthetic accuracy on
`hindi_natural.jsonl` / sibling training manifests; (c) train/val split
inside `LineDataset`.

**Why:** (a) is OOD by construction against #44's fixed training height,
so a synthetic-to-real gap would confound height mismatch with domain
shift. (b) is train-set memorization — `train.py` loads the full
manifest with no holdout — so the gap would look artificially small or
reversed. (c) is the cleaner long-term fix but costs a training-pipeline
change mid-queue; extending the export by 10 pages reuses resumable
export and leaves existing checkpoints valid.


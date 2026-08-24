# BOOK.md

## Preface

This is not a changelog. `DECISIONS.md` and `IMPLEMENTATION.md` already
cover what got built and why the code looks the way it does. This book
covers something else: the actual science underneath the code, explained
from first principles, chapter by chapter, in the order the repo got
built.

The test for every chapter is: could someone who has never opened this
repository, and doesn't know much about computer vision, read one
chapter and come away actually understanding both the idea and why this
project needed it? Not just "here's what we did" — "here's the problem
that exists in the world, here's why it's hard, and here's what building
this taught us about it."

Chapter 0 is written in full below. It's the template — read it before
writing any later chapter, both for tone and for how much ground a
chapter should cover.

---

## Chapter 0 — What Is Computer Vision, and Why Is Reading Hard For a Machine

### Start with what a computer actually sees

A photograph, to you, is a scene: a page of a book, a signature, a road
sign. To a computer, it is a grid of numbers. A single-color pixel is
three numbers — how much red, how much green, how much blue, usually
each from 0 to 255. A modest photo, say 1000 pixels wide and 1000 tall,
is three million numbers. There is no "page" in there, no "letter," no
"word." Just a grid of brightness values.

Computer vision is the field concerned with one question: how do you get
from that grid of numbers to something that counts as understanding —
"this says Priya," "this is a cat," "this door is open"? Every technique
in this repo, from a 30-million-parameter model to whatever Sarvam built
with a nine-figure budget, is ultimately an answer to that one question,
just at wildly different scales of ambition.

### Why "just teach it letters" doesn't work

The oldest and most tempting approach to reading text from an image is:
find each letter, look up what it is, done. This is called
segmentation-then-classification, and for a long time it's roughly how
OCR worked. It fails for reasons that are genuinely instructive:

- **Letters touch each other.** In cursive handwriting, in many fonts,
  and structurally in scripts like Devanagari (where a connecting line
  called the *shirorekha* runs across the top of a whole word), there
  often isn't a clean gap between one letter and the next to cut along.
- **One "letter" isn't always one visual unit.** In Devanagari, a base
  consonant can combine with a vowel sign (a *matra*) that appears
  above, below, beside, or wrapped around it, and two or more
  consonants can stack into a single fused glyph called a *conjunct*
  (a *saṃyuktākṣara*). The visually atomic unit — the thing a reader's
  eye treats as one character — is called a *grapheme cluster*, and it
  can correspond to two, three, or more separate values in the
  underlying digital text encoding. This mismatch between "one visual
  thing" and "one encoded thing" turns out to matter enormously for
  how you measure whether a system got the reading right — see
  Chapter 1.
- **Context changes what a shape means.** The same rough pixel pattern
  can be one letter in one font and a different letter in another; a
  smudge can look exactly like a diacritic. A system that classifies
  each cropped-out shape independently, with no knowledge of what came
  before or after it, throws away the single strongest signal a human
  reader actually uses: everything else on the page.

Modern systems, including the small one built in this repo, sidestep
segmentation almost entirely. Instead of "find each letter, then
classify it," the approach is "look at the whole image, and generate the
text one unit at a time, using both the image and everything generated
so far as context." That's a genuinely different kind of machine, and
it's worth understanding why it works.

### The two halves of a document-reading model

Every model in this repo, and Sarvam's, has (at least) two halves that
do fundamentally different jobs.

**The encoder** looks at the image and turns it into a set of numeric
*features* — vectors that capture "what's visually going on here,"
without yet committing to any specific letter or word. Modern encoders
for this job are usually Vision Transformers (ViTs): the image gets cut
into small square patches (say 14×14 pixels each), each patch becomes a
vector, and a mechanism called *self-attention* lets every patch's
representation get updated based on every other patch. This is what
lets the model notice, for instance, that a mark above a letter and the
base letter below it belong together as one grapheme cluster, even
though they're in different patches.

**The decoder** takes those image features and generates text, one
grapheme cluster at a time, left to right (or right to left, or however
the language's reading order actually runs — see Chapter 5). At each
step it looks at the image features and everything it has generated so
far, and predicts what comes next. This is the same basic mechanism
that powers large language models, just conditioned on an image instead
of, or in addition to, prior text.

The connective tissue between these two halves — how image features get
translated into something the text decoder can use — is itself a design
decision with real consequences, and it's one of the first things built
in this repo (Stage 2).

### Why this is a genuinely hard problem, not a solved one

English-language OCR on clean printed text is close to a solved problem
and has been for years. It is tempting to assume "OCR" in general is
solved and everything past that is refinement. This project exists
because that assumption is wrong for the majority of the world's
scripts, and the reasons are specific rather than vague:

1. **Data.** The internet, and therefore the training data every large
   model learns from, is overwhelmingly English and a handful of other
   high-resource languages. A model can only get good at reading what
   it has seen enough of.
2. **Script structure.** Complex scripts genuinely require more visual
   reasoning per character than Latin text does — conjuncts, stacking
   diacritics, and connecting strokes are not just "different letters,"
   they're a different geometric problem.
3. **These two causes get tangled together in every published result.**
   When a benchmark shows a language scoring low, that low score could
   be because the model barely saw that language during training, or
   because that script is intrinsically harder to read, or some mix of
   both — and a benchmark table alone cannot tell you which. Sarvam's
   own published results show exactly this pattern: a roughly 40-point
   spread across languages, with no explanation of how much is which
   cause.

That untangling — how much of a language's poor performance is "we
didn't show the model enough of it" versus "this script is inherently
harder to read" — is not answerable by looking at outputs alone. It
requires being able to control what a model sees during training and
then measure what it learned. That requires *owning* the model, not
querying it through an API. That single sentence is the entire reason
this project builds a model from scratch rather than only ever calling
someone else's.

### What's coming

Every chapter after this one follows the same shape: a real problem in
computer vision, why it's hard, and then what building a piece of this
repo taught about it. Chapter 1 starts with something that sounds
boring — how do you even *measure* whether OCR got something right? —
and shows why that question turns out to be far less obvious than it
looks, especially for scripts where "one character" and "one encoded
value" aren't the same thing.

---

## Chapter 1 — Measuring Correctness Is Its Own Hard Problem

*Status: not yet written. Will be written once Stage 0
(`src/eval/error_taxonomy.py`, `equivalence_tables.py`,
`transliteration_equivalence.py`) is implemented and has real results to
report.*

Preview of the angle: starts from "what does it even mean for OCR
output to be 'correct'," walks through edit-distance metrics (CER/WER)
and their hidden assumption that the reference text has one canonical
digital form, then shows — using this repo's own Stage 0 results —
what fraction of "errors" in real OCR evaluations turn out to be the
same text encoded two valid ways rather than genuine misreadings.

## Chapter 2 — Turning Text Back Into Pixels, On Purpose

A document-reading model is trained on pairs: an image of a page, and
the text that page is supposed to say. Most of the time those pairs
come from the wild — scans of books, photos of forms, screenshots of
websites — and the text side is whatever a human typed, or whatever an
older OCR system produced. That is fine when you want a model that
generalizes. It is the wrong kind of data when you want to *control*
what the model sees, so you can ask causal questions about why it fails.

This chapter is about the other direction: starting from text you
already know, and deliberately painting it into pixels. That sounds
backwards — why manufacture the thing you are trying to read? — but it
is the only way to hold every variable fixed except the one you want to
study. In this project the variable is *how often each visual character
appeared during training*. Everything else about Stage 1 exists to make
that dial real.

### What "drawing a letter" actually means for Indic scripts

If you have ever drawn Latin text on a screen, the naïve algorithm is:
look up a bitmap for `A`, place it, advance the pen by the width of
`A`, look up `B`, place it, and so on. That algorithm is already a
little wrong for Latin (kerning, ligatures like `fi`), and it is
hopeless for Devanagari or Bengali.

An Indic syllable — an *akshara* — is often several Unicode code points
that must be *shaped* into one visual glyph. A base consonant can take
a vowel sign (*matra*) that appears above, below, to the left, to the
right, or wrapped around it. Two or more consonants can fuse into a
conjunct (*saṃyuktākṣara*) whose shape is not the sum of its parts. The
shaping rules are font-specific and large. Hand-rolling them is a
career; this project does not try. It calls **HarfBuzz**, the same
shaping engine browsers and operating systems use, through the
`uharfbuzz` Python bindings.

HarfBuzz takes a string and a font and returns a sequence of glyph IDs
with advances and offsets. Each glyph is tagged with a *cluster* index
— the character offset in the original string it came from. That index
is how this repo attaches a pixel bounding box to a grapheme cluster:
group the glyphs whose cluster falls inside one grapheme, union their
advances, and you have a box around the visual unit the instrument will
later treat as one vocabulary token. Painting is delegated to Pillow
(with raqm/HarfBuzz under the hood) so the pixels and the boxes agree
on what was drawn.

The code that does this lives in `src/renderer/render.py`. A smoke page
of Hindi from GlotOCR ground truth renders in well under a second
(roughly 40–80 ms on a laptop for a short page; the Stage 1 acceptance
bar is one second). The output is a PNG plus a JSON file listing every
grapheme cluster and its `[x0, y0, x1, y1]` box.

### Layouts from real documents, not invented frames

A page is not only characters. It has columns, margins, headers,
tables, form fields. Inventing a "two-column template" by hand is easy
and wrong: systems that look identical on clean synthetic pages fall
apart on real ones, and a benchmark built on invented geometry would
overstate every model in this repo. So Stage 1 pulls geometry from
real sources — Internet Archive scanned Indic books (via IIIF page
images), government PDFs, Wikipedia Indic articles as printable PDFs —
and stores them as a bank of region boxes under
`data/cache/layouts/bank.json`. Born-digital PDFs donate their text
and table boxes directly; camera scans get columns inferred from ink
projection profiles. The classifier labels each page
`single-column`, `two-column`, `marginalia`, `table-embedded`, or
`form` — the same axis Stage 3 will use for reading-order difficulty.

That work is `src/renderer/layout_sources.py`. It is deliberately not
a neural layout detector. Stage 1 has to be deterministic and
inspectable; a second model’s mistakes would leak into Probe 1.

### Degradation as a measured distribution

Real paper is blurry, noisy, crooked, and sometimes translucent enough
that the reverse side ghosts through. Hardcoding
`GaussianBlur(radius=1.2)` would make the "degraded" condition a
fiction. Instead, `src/renderer/degradation_profile.py` measures four
apply-parameters — blur sigma, noise std, skew degrees, show-through
alpha — off real scans (1600px-wide Internet Archive pages, plus a
heavier GlotOCR "old document" tail for noise and show-through that
clean DLI JPEGs lack), and stores them as an empirical joint
distribution. Sampling draws a whole page’s four-tuple, so blur and
noise stay correlated the way they were on paper.

Blur is inverted from Laplacian variance against a calibration curve
built by blurring a Wikipedia-print page rasterized at a matching
scale. Measuring against line-level synthetic images first produced
sigma≈0 on every real book page — a scale mismatch, logged and fixed
in `DECISIONS.md` #24.

### The dial: glyph-frequency mode

Probe 1 needs three training conditions with identical data *volume*
and different exposure distributions:

- **natural** — the corpus as it is
- **flattened** — push toward uniform over observed grapheme clusters
- **inverted** — rare clusters inherit the mass of common ones

`src/renderer/glyph_frequency.py` implements that dial. Frequency is
counted at the grapheme-cluster level (not code points — Chapter 1’s
lesson), and only for clusters that contain Indic script characters,
so the dial cannot waste itself promoting a stray `%`. Resampling is
mostly sentence-level importance sampling: whole sentences are redrawn
with weights that favour those whose cluster mix is closer to the
target. That choice is about the naturalness confound
(`DECISIONS.md` #10) — if you scatter rare conjuncts into random
positions, you have also made the text less like language, and you can
no longer tell whether the model failed to *see* the glyph or failed
to *expect* it. Probe 3 (blank images) exists to measure the guessing
half of that story later.

On a 60-line GlotOCR Hindi slice, inverted and flattened cannot hit
their targets tightly — every sentence is already a natural mix — but
the *direction* of the dial is verified, and a constructed toy corpus
hits the acceptance checks. Probe 1 will feed a larger rendered
corpus through the same code.

### Three realism tiers on one renderer

The same `render.py` entry points serve three jobs:

- **Tier A** — fixed font, clean (zero) degradation, only
  glyph-frequency mode varies. Probe 1’s causal condition.
- **Tier B** — layout, font, and degradation sampled from the measured
  pools. Headroom for Probes 2–5.
- **Tier C** — unmodified real documents with existing transcriptions.
  Probe 6’s reality check; no synthetic boxes.

A verification run of 100 Tier A pages (plus sample Tier B/C) wrote
under `data/cache/renders/verify_100/`: every page under 350 ms, mean
about 82 ms; natural-mode total variation to target is zero (exact
multiset); Tier B pages visibly pick up measured blur and paper tone.

### What this stage taught

Owning the renderer is what makes "exposure versus complexity" a
question you can ask rather than a story you tell after the fact. The
surprises were mundane and useful: layout classification must ignore
tiny Wikipedia infobox tables or every article becomes
`table-embedded`; blur calibration must match pixel scale or every
real scan looks "perfectly sharp"; the frequency dial must ignore
Latin punctuation or inverted mode optimizes the wrong alphabet. None
of those show up in a design doc until the first real page is measured.
They are now in `DECISIONS.md` so the next session does not re-learn
them.

Stage 2 starts from these `(image, cluster-GT)` pairs and trains the
instrument that the probes will open up.

## Chapter 3 — How a Machine Learns to Read, Starting From Nothing

*Status: not yet written. Will be written once Stage 2a (the instrument)
is implemented and has trained at least once.*

Preview: patch embeddings and self-attention made concrete with this
repo's actual encoder, autoregressive generation and why it's different
from classification, tokenization choices and why this project uses
grapheme clusters instead of the more common byte-pair encoding.

## Chapter 4 — Teaching an Existing Model a New Trick Cheaply

*Status: not yet written. Will be written once Stage 2b (the demo,
LoRA-adapted) is implemented.*

Preview: what it means for a model to already "know" something from
pretraining, why fine-tuning the whole thing is often wasteful, and how
LoRA lets you adapt a large model by training a small number of extra
parameters instead of touching the original weights.

## Chapter 5 — Where Does the Text Go on the Page

*Status: not yet written. Will be written once Stage 3 (layout and
reading-order modules) is implemented.*

Preview: why "top to bottom, left to right" breaks on real documents,
reading order as a permutation problem rather than a classification
problem, and why Kendall tau — not accuracy — is the right way to score
it.

## Chapter 6 — Reinforcement Learning, From Scratch

*Status: not yet written. Will be written once Stage 4 (RLVR) is
implemented.*

This chapter must open, per the plan for this book, by explaining what
reinforcement learning actually is — before any mention of this
project's specific use of it. At minimum it should cover: the difference
between learning from a fixed correct answer (supervised learning, what
the SFT stage does) and learning from a reward signal computed after
the fact (reinforcement learning); what makes a reward "verifiable";
why a poorly designed reward gets *gamed* rather than optimized in the
intended sense, using this project's own coverage-term ablation as the
concrete, observed example rather than a hypothetical one; and only then,
the specific sequence of steps this repo takes to retrain the demo model
with RLVR on top of its SFT checkpoint.

## Chapter 7 — What Does the Model Actually Know

*Status: not yet written. Will be written once Stage 5's probes
(exposure/complexity, confusion structure, blank-image control,
calibration) produce real results.*

Preview: the difference between a model being *right* and a model
*knowing* it's right (calibration), why a model can be confidently
wrong on things it has barely seen, and what this repo's exposure
experiment actually showed once the numbers came in — including if the
result was messy or didn't show a clean effect, because that's still
worth explaining honestly.

## Chapter 8 — Why You Can't Learn Everything From an API

*Status: not yet written. Will be written once Stage 5's Sarvam transfer
analysis is complete.*

Preview: what a closed API does and doesn't expose (outputs, yes; the
training data, the internal confidence, the raw output distribution,
no), why that forecloses certain kinds of questions entirely, and what
did and didn't transfer when this project's own findings were checked
against Sarvam's real, production-scale system.

## Chapter 9 — Deciding When to Trust a Machine and When to Ask for Help

*Status: not yet written. Will be written once Stage 6 (the triage
cascade) is implemented.*

Preview: selective prediction and the idea of "abstaining" instead of
guessing, why a confidence score is only useful if it's calibrated,
and how this repo's cascade experiment measured whether that was true
here — including how it's judged (router quality) rather than what it
might naively seem to promise (cost savings), and why that distinction
matters.

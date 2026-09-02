# AGENTS.md

Instructions for any coding agent (Cursor, Claude Code, or otherwise)
working in this repository. Read this file first, every session.

## Read order, every session

1. `AGENTS.md` (this file)
2. `IMPLEMENTATION.md` — find the first unchecked `[ ]` item in stage
   order. That's the next task, unless the person tells you otherwise.
3. `DECISIONS.md` — skim for anything relevant to the module you're
   about to touch, so you don't re-litigate a settled choice.
4. `TODO.md` — confirm the task you're about to do is next in sequence,
   not out of order.

Do not start writing code before doing this. The whole point of these
docs is that you shouldn't need the full prior conversation history to
know what to build next — the repo state should be self-describing.

## The workflow for every unit of work

1. Pick the next unchecked item in `IMPLEMENTATION.md`, in stage order.
   Don't skip ahead to a later stage because it looks more interesting.
2. Implement it.
3. Write tests where the module has a checkable property (metric
   functions, renderer output, equivalence tables) — not for everything,
   but anything with a clear right answer gets a test.
4. Run it. Do not report a result you have not actually run. If you
   can't run something (no GPU available in this session, etc.), say so
   explicitly and mark the item `[!]` blocked with a one-line reason,
   don't mark it `[x]`.
5. Flip the status in `IMPLEMENTATION.md`.
6. If you made any non-obvious choice along the way — a library,
   a threshold, a scope cut, a metric definition — append an entry to
   `DECISIONS.md`. Follow the existing format: Decision / Alternatives
   considered / Why. Number it sequentially. Don't rewrite past entries.
7. Update `TODO.md` to reflect what's now done.
8. If the unit of work completed a conceptually whole piece (a full
   stage, or a probe, or a clearly self-contained sub-part), write or
   extend the matching chapter in `BOOK.md`. See the "Writing BOOK.md"
   section below for what that means and how to do it well.

## Code style

- **Every function gets a docstring that explains why it exists and
  where it sits in the pipeline, not just what it does.** The person
  using this repo wants to open any function months later and
  immediately understand: what is this for, why does it do things this
  way, what calls it, what does it hand off to next. A docstring that
  just restates the function signature in English is not sufficient.

  Bad:

  ```python
  def compute_tau(pred_order, gt_order):
      """Computes Kendall tau between predicted and ground truth order."""
  ```

  Good:

  ```python
  def compute_tau(pred_order, gt_order):
      """
      Kendall tau between predicted and ground-truth block reading order.

      Why tau and not accuracy: reading order is a permutation problem,
      not a classification problem. Accuracy would treat "swapped two
      adjacent blocks" the same as "read the page in reverse," which
      isn't useful signal. Tau captures how far off the ordering is,
      not just whether it's exactly right.

      Called from: eval/reading_order_metric.py, once per document, per
      layout-complexity bucket (see IMPLEMENTATION.md Stage 3). Feeds
      into the tau-vs-complexity curve that tests whether the demo
      model's separate reading-order module is carrying real load.
      """
  ```

- Comment non-obvious lines inline, not just at the function level —
  especially anything touching Unicode normalization, grapheme
  clustering, or glyph-frequency resampling, where the "obvious" reading
  of the code is often wrong for Indic scripts specifically.
- No dead code, no commented-out experiments left in place — if an
  approach was tried and abandoned, that belongs in `DECISIONS.md` as a
  rejected alternative, not as a code comment.
- Every script that will run on Colab assumes the session can die
  mid-run. Checkpoint by default. Make resumption the normal path, not
  a special case.

## Long-running scripts: progress and resumability are not optional

Any script that processes more than a handful of items — a batch of
images, a training loop, a set of API calls — must do both of these,
not just for Colab training runs (which IMPLEMENTATION.md already
requires this of) but for **any** script that could plausibly run for
more than a minute or two:

- **Print progress per item, not just per batch.** A script that prints
  "120 images to process" and then goes silent until "done" is
  indistinguishable from a hung process. Print something on a
  reasonable cadence — every N items, or every image if the per-image
  cost is high (as with Surya-style recognition models) — so whoever's
  watching the terminal can tell "slow" from "stuck" without having to
  go check `ps aux` in another tab.
- **Checkpoint and resume by default.** If the script dies partway —
  crash, Ctrl+C, laptop sleeps — re-running it should pick up from
  where it left off, not silently redo finished work or silently
  overwrite a partial output file. Concretely: write results
  incrementally (append, don't buffer everything in memory until the
  end), and on startup, check what's already been written and skip
  it. This applies to `run_baselines.py`-style batch scripts just as
  much as it applies to model training.

This came up directly after `run_baselines.py` ran with no per-image
output for over ten minutes and looked identical to a hang. Don't wait
for that to happen again before adding this — treat it as a standard
requirement on any new batch-processing script from here on.

## Heavy scripts run on Colab, not the local machine

OCR batch runs, model training, and anything else expected to take more
than a few minutes is written to run cleanly on Google Colab (free T4).
Do not assume a local GPU, a long-lived local process, or paths that
only exist on one laptop. The laptop checkout is where results _land_,
not where the heavy work has to execute.

Concretely, every such script:

- **No hardcoded local-only paths.** `/Users/...`, home directories,
  and machine-specific caches do not belong in the code. Relative
  paths under the data root are fine; an absolute path is a Colab
  `--data-root` (or Drive mount) argument, not a constant.
- **One configurable root for inputs and outputs.** Switching machine
  (laptop → Colab, Colab → Drive) is one path: `--data-root` or
  `OCR_DATA_ROOT`. Raw inputs and predictions both hang off that root
  (`{root}/raw`, `{root}/predictions`), matching this repo's `data/`
  layout, so it is just one path to change. A nested override for an
  isolated smoke test is allowed (`--pred-root` / `OCR_PRED_ROOT`) so
  a test never writes into a live run's tree — that is not a second
  Colab knob.
- **An explicit export cell/step at the end.** Colab disks vanish
  when the session dies. The last cell/step zips or copies results
  somewhere downloadable (`google.colab.files.download()`, or a Drive
  mount) and names the files so they unpack into this repo at the
  path `IMPLEMENTATION.md` already specified for that stage. Do not
  invent a second Colab-only output layout. For Stage 0 baselines
  that path is `data/predictions/{engine}/{language}.jsonl`.
- **Probe result jsonl files: commit and push immediately.** Files
  under `data/probe_results/` are small and must land in git right
  after generation — before starting any other Colab run. Do not treat
  them like `data/predictions/` (heavy, regenerable, gitignored). If
  `git push` is rejected for divergent branches, resolve it (`git pull
  --rebase`, then push) before moving on; unpushed Colab probe results
  have already been lost once when a fresh clone wiped them.

The long-running-scripts rules above still apply — Colab sessions
die, which is why those rules exist. Checkpoint into the data root;
export is how the checkpoint gets back here.

## Hard constraints, repeat from IMPLEMENTATION.md because they're easy to forget

- Free Colab T4 only. Turing architecture — no `bf16`, no
  FlashAttention-2. Use `fp16` and gradient checkpointing.
- Sarvam API budget is ~200 pages total, ever, for this entire project.
  Never call `sarvam_client.py` in a loop without checking
  `data/cache/` first. If you're about to write code that calls the
  Sarvam API more than once per unique page, stop and re-read
  `DECISIONS.md` #17.
- Do not fabricate or recall-from-training-data any fact about Sarvam's
  current product, pricing, or published benchmark numbers. Sarvam
  ships fast; the project's own original research pass is already
  known to be stale by design. If a task needs a current Sarvam fact,
  fetch it fresh (docs.sarvam.ai, their GitHub, their blog) rather than
  assuming last-known state is still true.
- No invented citations, no invented benchmark numbers from other
  papers. If a claim needs literature support, it needs an actual
  fetched source, not a plausible-sounding recollection.

## Writing BOOK.md

`BOOK.md` is not a changelog and not a second copy of `IMPLEMENTATION.md`.
It's a standalone explanation of the computer vision (and, where
relevant, reinforcement learning) underneath whatever was just built,
written so a beginner who has never seen this repo could read one
chapter and understand both the concept and why this project needed it.

When you finish a stage or a probe:

1. Start from the first-principles concept, not from the code. If you
   just built the reading-order module, the chapter opens with "what is
   reading order, why is it a hard problem for a machine, why doesn't
   left-to-right/top-to-bottom work" — before it ever mentions this
   repo's specific implementation.
2. Connect the concept to what got built, concretely, referencing real
   file paths and real numbers/plots produced by the code, not
   hypothetical ones.
3. State what was learned or found, in plain language, including
   negative or messy results — a probe that didn't show a clean signal
   is still worth a paragraph explaining why, that's part of the
   education.
4. Keep the chapter self-contained. Someone should be able to read
   Chapter 4 without having read Chapter 3, even if it costs a
   sentence or two of repeated context.
5. Match the tone and depth of the existing chapters — read at least
   the two most recent chapters before writing a new one, so the book
   doesn't shift register halfway through.

Chapter 0 in `BOOK.md` is already written as the template for this. Read
it before writing Chapter 1.

## When something in this conversation contradicts the docs

The docs win. If a person's message in a chat session seems to conflict
with `DECISIONS.md`, ask before overriding a recorded decision — don't
silently deviate from something that was deliberately chosen and
justified in writing.

## Colab execution log

Track every script run on Colab here, so it's clear what's been executed
remotely vs locally.

| Script                                      | Purpose                                           | Output location                        | Status      |
| ------------------------------------------- | ------------------------------------------------- | -------------------------------------- | ----------- |
| src/eval/run_baselines.py                   | Surya/PaddleOCR completion for Santhali/Kashmiri  | data/predictions/ (confirm exact path) | in progress |
| src/data_pipeline/export_manifest_scaled.py | Real Hindi/Bengali line manifests, 100 pages/mode | data/manifests/                        | in progress |

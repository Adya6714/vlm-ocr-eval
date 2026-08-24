# ocr-vlm-eval

A small document VLM, built the way Sarvam built theirs — encoder, projector,
language model, plus separate layout and reading-order modules, trained SFT
then RLVR — instrumented so a set of diagnostic probes can run on it that a
closed API forecloses. The same probes then run against Sarvam Vision, to
see what transfers.

## Why this exists

Sarvam published a per-language OCR table with a roughly 40-point spread —
Hindi at 95.91, Kashmiri at 55.93 — and no explanation for the gap. Anyone
can read that table. Explaining it requires opening a model: exposure,
logits, and confidence are invisible behind an API. So this repo builds a
small one, develops a diagnostic protocol on it, and runs as much of that
protocol against Sarvam's real API as a free-tier budget allows.

This is not a claim to have found something Sarvam doesn't already know
internally. It's a reproducible, causal decomposition of something
currently only knowable from inside the building — plus a set of tools
that would let _any_ team ask the same questions of _any_ multilingual OCR
model, to know where to push on the next training run.

## What's in this repo, and where to look

This repo is documentation-first on purpose — the docs are what a coding
agent (and you, later) checks against to know what's built and why.

| File                                       | What it's for                                                                                                                                                                                                                                 |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) | The technical spec. Every module, its inputs/outputs, its acceptance criteria, and its current status. This is what an agent verifies progress against before starting new work.                                                              |
| [`DECISIONS.md`](./DECISIONS.md)           | A running log of _why_ — every non-obvious architectural or scope choice, with the alternatives that were considered and rejected. Read this when a piece of code looks like an odd choice.                                                   |
| [`AGENTS.md`](./AGENTS.md)                 | Instructions for any coding agent (Cursor, Claude Code) working in this repo — how to build, how to comment, how to update the other docs.                                                                                                    |
| [`TODO.md`](./TODO.md)                     | The sequenced task list, stage by stage.                                                                                                                                                                                                      |
| [`BOOK.md`](./BOOK.md)                     | A first-principles explanation of the computer vision (and reinforcement learning) underneath this repo, written chapter by chapter as each stage is built. Start here if you want to understand _why the code works_, not just what it does. |

## Project shape, in one paragraph

Two models get built, deliberately kept separate. **The instrument** is a
small vision-language model trained entirely from scratch — it has seen
zero Indic text before this project touches it, which is what makes it
possible to _control_ exposure and run causal experiments. **The demo** is
a LoRA-adapted small open VLM with the same component shape Sarvam uses —
encoder, projector, LM, a separate layout module, a separate reading-order
module, SFT then RLVR — which exists to prove the production architecture
can be built, not to generate findings. Six diagnostic probes run on the
instrument; as many as an API permits run against Sarvam Vision for
transfer.

## Status

Stage 0 scoring (Tier 1/2 + `error_taxonomy.py`) has been run on real
engine predictions; Surya/Paddle full-corpus OCR and UNREVIEWED labels
are still open. Stage 1 renderer is verified. Stage 2a instrument +
Probe 1 orchestration exist but are smoke-only. To verify the whole
pipeline without real data or compute: `make smoke-test` (Makefile
target `smoke-test`; see `TODO.md` for current smoke-path blockers).

## Compute and budget constraints (hard, non-negotiable)

- Zero budget. Sarvam's ₹100 free signup credit (~200 pages at ₹0.5/page,
  max 10 pages/job) is the only paid usage anywhere in this project.
- Free Colab T4 only (~16GB, session timeouts, no persistent storage,
  Turing architecture — no bf16, no FlashAttention-2).
- No frontier API baselines (Gemini/GPT) unless separately free.

## Quickstart

```bash
make smoke-test
```

(`makefile` phony target is `smoke-test`.) That is the no-data, no-GPU
check of Stage 0 tier tests, the instrument `__main__` smokes, fake
Probe 1 (9 short runs), and `generate.py`. It does not produce a
research finding. Real OCR / training still follows `TODO.md` and
Colab (`AGENTS.md`).

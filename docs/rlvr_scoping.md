# RLVR real retrain — Phase 1 scoping (investigation only)

**Date:** 2026-09-13  
**Status:** no training code written; no GPU / Colab this pass.  
**Question:** what exists, what Chapter 6 actually asked for, and how
large a Colab task a coverage-term-removed **retrain** would be.

Companion that already records the last Colab outcome:
`docs/tier2_rlvr_ablation.md` (SFT greedy scored with λ=0; no policy
update).

---

## Short answers

| # | Question | Answer |
|---|---|---|
| 1 | Algorithm in BOOK Chapter 6 | **Not specified.** The chapter specifies a *reward* and a *one-ablation experiment*, not PPO / REINFORCE / rejection sampling. “Cheap” refers to **one retrain**, not a lighter optimizer. |
| 2 | What `rlvr_train.py` contains | Reward **scoring** on greedy decode. **No** training rollouts (no logprobs, `no_grad`), **no** policy update. The generate path is eval-only and likely mis-wired vs SFT. |
| 3 | Why `generate()` “did not complete” | **There is no such gate.** Retrain is unconditionally unimplemented. The “unless generate() works and session time remains” sentence is a **static JSON note**, not a branch that was tested. |
| 4 | Minimal work for the book ablation | **New RL (or RS) infrastructure**, not a small patch on an existing loop. Honest size: a real training script, not ~N lines of PPO clip on top of ready rollouts. |

---

## 1. What BOOK.md Chapter 6 actually specifies

Read from `BOOK.md` “## Chapter 6 — Reinforcement Learning, From Scratch”
through the chapter close (before Chapter 7). Not from an earlier chat
summary.

### Training procedure (algorithm)

The chapter describes RL at the **setup** level:

```text
Image → Model → complete output → evaluate the whole reading → REWARD
→ model learns to maximize reward
```

It contrasts that with token-level SFT. It does **not** name:

- PPO (clipped surrogate, critic, GAE, multiple epochs)
- GRPO
- REINFORCE / vanilla policy gradient
- best-of-n / rejection sampling

“RLVR” is defined as **Reinforcement Learning with Verifiable Rewards**
(automatic scores from renderer GT), not as a particular optimizer.

The only named update language is “the model learns to maximize
reward.” Decision #82 (not the chapter) is the first place the repo
names **PPO/GRPO** as the deferred trainer.

The module docstring on `src/models/demo/rlvr_train.py` claims
“This is REINFORCE on sequence logprob × reward.” That is **not** in
Chapter 6, and the file body does not implement it. Treat the docstring
as an unimplemented intent, not as the book spec.

### What “one cheap ablation” means

It is **not** “use a cheap algorithm because the model is 19.5M.”

Chapter 6:

- Does not want a sweep across every reward term.
- Wants **one** comparison (Decision #11): full reward vs **remove
  coverage** (`λ=0`), one retrain, then ask whether the policy starts
  omitting difficult content.

“Cheap” = **one retrain**, “a mechanism, not a spreadsheet of tiny
deltas.” Same wording as `DECISIONS.md` #11.

The **19.5M** figure in `IMPLEMENTATION.md` Stage 2a is the
**from-scratch instrument** (`|V|=367`), not the RLVR target. Chapter 6
places RLVR **after demo SFT**. The trained demo checkpoint is
`ds4sd/SmolDocling-256M-preview` LoRA (`docs/tier2_stage2b_demo.md`),
hundreds of millions of base parameters, not 19.5M. Do not size the
Colab task as “tiny model, so REINFORCE is obviously enough.”

### Planned reward (this *is* specified)

```text
R = character accuracy + TEDS + reading-order quality − coverage penalty
```

In code (`src/models/demo/rlvr.py`):

```text
R = char_acc + teds + tau - lambda_coverage * (1 - coverage)
```

`lambda_coverage=0` is the ablation hook. TEDS and tau are **inert
(0)** unless the caller passes table trees / block orders. The SFT /
RLVR path currently uses **line-crop** GT (`LineCropDataset`), so those
two terms are always 0 in the existing train script.

### What the chapter says is still missing

SFT exists (100 LoRA steps). The coverage-term **retrain** has not run.
Cell 10 scored SFT greedy decode with λ=0 and did not update the
policy. The chapter calls itself a **design / blueprint** until someone
inherits: verifiable reward, known gaming mode, one ablation.

There is a **stale sentence** at the end of the chapter: “there is no
SFT adapter yet.” That contradicts the same chapter’s earlier “SFT for
the demo now exists” and `docs/tier2_stage2b_demo.md`. The adapter
lived on Colab Drive, not this git tree; the scientific gap is the
**retrain**, not the missing SFT.

---

## 2. What `rlvr_train.py` currently contains

File: `src/models/demo/rlvr_train.py` (119 lines). Reward math lives in
`src/models/demo/rlvr.py`. Tests: `tests/test_demo_rlvr_order.py`
(reward **shape** only).

### Control flow (actual)

1. Refuse if `--device cuda` and no CUDA (`SystemExit`, no fake laptop
   result).
2. If `adapter_config.json` missing under `--sft-root`, write
   `rlvr_not_trained.json` and exit 0.
3. If adapter exists:
   - Print: would train λ from SFT for `max_steps`; **“Full PPO not
     implemented”**; write an omission diagnostic on **greedy decode**
     instead.
   - Load PEFT adapter, `model.eval()`, `torch.no_grad()`.
   - For `n = min(32, len(dataset))` line crops: `processor(images=…)`
     then `model.generate(..., max_new_tokens=64)`.
   - On any exception: `hyp = ""`, print the error, **continue**.
   - `compute_reward(hyp, gt, lambda_coverage=args.lambda_coverage)`
     and again at λ=0. Default `teds=0`, `tau=0`.
   - Write `rlvr_omission_sample.jsonl` + `rlvr_summary.json`.

`--max-steps` (default 30) is parsed and printed and **never used**.
There is no optimizer, no `model.train()`, no backward, no adapter
save under `checkpoints/demo_rlvr_nocov`.

### Rollouts vs update — the distinction that matters for effort

| Piece | Present? | Notes |
|---|---|---|
| Reward function `compute_reward` | **Yes** | Including λ=0. |
| Unit tests for omission **shape** | **Yes** | Uses `use_emitted_only_acc=True`. The train script does **not**. |
| Sampling / training rollouts | **No** | Greedy `generate` under `no_grad`; no token logprobs; exceptions become empty hyp. |
| Policy-gradient / PPO update | **No** | Explicitly refused in the print; loop never started. |
| Checkpoint / resume for RL | **No** | SFT has this; RLVR train does not. |
| Chat-template prompt matching SFT | **No** | See below. |

So this is **not** “has rollouts, missing the update rule.” It is
**has a scalar and an eval decode**, missing **both** a training-time
rollout (log π(a|s) for sampled tokens) **and** an update.

### Generate wiring vs SFT (why the eval loop is not reusable as-is)

`sft.py` `_encode_example` uses `apply_chat_template` with an image +
“Transcribe the text in this image.” + assistant GT.

`rlvr_train.py` calls `processor(images=item["image"], return_tensors="pt")`
with **no text / no chat template**. For SmolDocling (Idefics3-style),
that is a different input than the model was SFT’d on. Even a
non-throwing `generate()` is not a drop-in training rollout.

### Last Colab numbers (not re-run here)

From `docs/tier2_rlvr_ablation.md`: the **corrected** SFT-greedy baseline
is n=32, mean_coverage ≈ **0.309**, 32/32 below 0.5, with `n_empty_hyp=0`
and zero swallowed exceptions (Decision #88). This is **not** a λ=0-trained
policy; it is the baseline any later retrain must be compared against.

---

## 3. Why `generate()` “did not complete” last time

The earlier phrase — retrain “skipped unless `generate()` works and
session time remains” — is **copy in the summary JSON**, not a
condition that ran.

Exact text written every successful diagnostic run
(`rlvr_train.py` `summary["note"]`):

> This is SFT-greedy scored with λ=0 reward, not a retrained policy.
> Retrain-from-SFT is skipped unless generate() works and session time
> remains for a real RL loop.

**There is no `if` on generate success. There is no session-time /
wall-clock check.** After the adapter-exists branch, the script always
does the 32-example decode and always writes that note. PPO is skipped
because it **is not in the file**.

What *does* exist around generate:

```python
try:
    ...
    gen = model.generate(**inputs, max_new_tokens=64)
    hyp = processor.batch_decode(...)[0]
except Exception as e:
    hyp = ""
    print(f"[rlvr] generate failed: {type(e).__name__}: {e}")
```

That is **per-example swallow**, not “abort retrain.” Retrain is
already not attempted.

Cell 10 in `scripts/build_colab_run_nb.py` only skips RLVR if
`checkpoints/demo/adapter_config.json` is missing. Last session **had**
the adapter (`docs/tier2_stage2b_demo.md`), ran `rlvr_train.py`, and
got `rlvr_summary.json`. That is a completed **diagnostic** path, not
a timed-out PPO job and not a generate-success gate that failed.

**What was actually detected:** “PPO not implemented; write greedy
scores.” **What was not detected:** generate OK vs generate crash vs
empty decode vs leftover session minutes. `docs/tier2_rlvr_ablation.md`
already says Cell 10 stdout was not pasted.

---

## 4. Minimal implementation for the book ablation

### What the book actually needs as an *outcome*

From an existing SFT adapter:

1. Retrain **once** with coverage **off** (`λ=0`).
2. Compare **coverage** (and omission of hard content) against the
   **SFT baseline**.
3. Optionally a second run with λ>0 if you want the “normal RLVR”
   arm; Decision #11’s checkable claim is the **removed** term.

That requires the policy to **change** under a reward that can
**prefer omission**. Scoring SFT with λ=0 does not do that.

### Scientific hole (must be in the Colab spec, not only the trainer)

Chapter 6’s gaming story: accuracy on **emitted** text can go up if
you drop hard spans. The unit test `ablation_omission_signal` uses
`use_emitted_only_acc=True`.

`compute_reward` **defaults** to `char_accuracy`, which is
Levenshtein against **full GT** and therefore **already penalizes
deletions**. `rlvr_train.py` uses that default.

If a future loop optimizes default `char_acc + teds + tau` at λ=0,
omitting the tail is **not** the same cheat as in Chapter 6 / the
unit test. A λ=0 retrain could fail to show omission **even with a
correct REINFORCE loop**, because the accuracy term is not the
gaming channel the book describes.

**Minimal scientifically honest ablation** therefore needs, in addition
to an update rule:

- `use_emitted_only_acc=True` (or an equivalent precision-style term)
  on the λ=0 arm, **or** a documented decision that default
  `char_acc` is enough and the book’s example is only pedagogical.
- Line-crop data ⇒ TEDS/tau stay 0. That is acceptable for a
  **coverage** ablation; it is not the full three-term document
  reward in the chapter diagram.
- SFT baseline coverage in the last diagnostic is **0**. A λ=0
  retrain cannot “learn to omit” relative to a model that already
  covers nothing on this decode path. Fix generate (chat template,
  same collate as SFT) and re-measure SFT coverage **before**
  claiming an RL effect. 100-step smoke SFT may still be too weak
  for the omission story (already flagged in
  `docs/tier2_rlvr_ablation.md`).

### Concrete missing pieces (not “add PPO clip to existing rollouts”)

**Reusable today**

- `compute_reward` / coverage / tests (~160 lines of reward, done).
- SFT LoRA train skeleton: PEFT load/save, CUDA refusal, line-crop
  dataset, fp16-ish device move, per-step print, periodic
  `save_pretrained` (`sft.py`).
- Colab Cell 10 already invokes this script with λ=0.

**Must be built**

1. **Generation that matches SFT** (chat template + image), sampling
   (`do_sample=True` or multinomial from logits), not greedy-only
   eval.
2. **Logprobs of the sampled continuation** (forward or
   `generate(output_scores=True)` / teacher-force the sampled ids).
   The current `no_grad` `generate` throws those away even when it
   works.
3. **An update.** Closest to the unused `rlvr_train.py` docstring:
   REINFORCE, `loss = -(R - baseline) * sum log π(tokens)`, one
   sequence per step, 30 steps as already flagged. Roughly
   **150–300 lines** if you copy SFT’s loop and do **not** add a
   critic. Full PPO (value head, GAE, clip, minibatch reuse) is
   **substantially more** and is **not** required by Chapter 6.
4. **Resume / per-step progress** (`AGENTS.md` batch-script rules).
5. **Eval pass:** same decode on a held slice for SFT vs RLVR-nocov
   mean coverage — the actual Decision #11 number.

There is **no** existing rollout buffer to hang an update on.
Calling this “~40 lines of policy gradient on the current loop” would
be false.

### Size judgment for the deadline

| Option | Fits “small addition”? | Enough for Chapter 6’s *claim*? |
|---|---|---|
| Full PPO/GRPO on SmolDocling | No. New stack. | Yes if reward + eval are right; cost is the problem. |
| REINFORCE (docstring intent) | **No — new loop**, but bounded: one file, ~one Colab session if generate is fixed first. | Yes **if** the accuracy term is the gaming one and SFT coverage is not already ~0. |
| Best-of-n rejection sampling: sample k, pick max R, SFT on winners | Smaller than PPO; still needs working sample+reward+SFT step. | Tests “does maximizing this R drop coverage?” without a policy-gradient estimator. Weaker as “RL,” closer to the **mechanism** the chapter cares about. |
| Keep scoring SFT with λ=0 | Already done | **No.** |

**Plain recommendation for Phase 2:** treat this as **substantial new
development relative to the repo today**, not a one-cell completion of
a 90%-written trainer. If the submission deadline cannot absorb a
working sample+logprob+update (or RS) **and** a generate fix **and** a
non-zero SFT coverage baseline, defer the retrain; the paper/book
already describe the design. If one more T4 session is available,
**do not start with PPO**. Specify either:

- **REINFORCE**, λ=0, `use_emitted_only_acc=True`, chat-template
  generate, 30 steps from `checkpoints/demo`, compare coverage to
  SFT on the same decode, or
- **rejection sampling / best-of-n** against that same R, then a
  short SFT on winners — fewer moving parts, still a trained-policy
  contrast.

Either is a **new Colab task**, not “uncomment the PPO branch.”

---

## Pointers

| File | Role |
|---|---|
| `BOOK.md` Chapter 6 | Reward + one ablation; no optimizer name |
| `DECISIONS.md` #11 | Ablation = coverage term only |
| `DECISIONS.md` #82 | Ship reward; do not start PPO/GRPO until SFT exists (SFT now exists; PPO still absent) |
| `src/models/demo/rlvr.py` | `R`; no trainer |
| `src/models/demo/rlvr_train.py` | Diagnostic greedy decode; PPO/REINFORCE not implemented |
| `src/models/demo/sft.py` | Working LoRA loop + chat collate to copy |
| `docs/tier2_rlvr_ablation.md` | Corrected baseline: n=32, mean_coverage≈0.309, 32/32 < 0.5; no retrain |
| `scripts/build_colab_run_nb.py` Cell 10 | Invokes the diagnostic script |

Phase 2 Colab instructions are intentionally **not** in this file;
they depend on choosing REINFORCE vs rejection sampling vs defer.

---

## Phase 2a (2026-09-13) — generate fix, no trainer

Fixes in `src/models/demo/sft.py` and `src/models/demo/rlvr_train.py`.
**No rollout / update-rule code.** Decision #88.

### 1. Chat template now matches SFT

SFT `_encode_example` (the collate that actually ran 100 LoRA steps)
uses:

- user: image + text `"Transcribe the text in this image."`
- assistant: GT
- `apply_chat_template(..., add_generation_prompt=False)`

That user turn is now `sft_user_turn()` / `SFT_USER_INSTRUCTION`.
Generate uses the **same user turn** with `add_generation_prompt=True`
(`encode_for_generate`). Continuation tokens only
(`decode_continuation`); the old path `processor(images=...)` with no
text is refused (`RuntimeError` if there is no chat template).

### 2. Generate failures are no longer `hyp=""`

The `except Exception: hyp = ""` branch is gone. A failed `generate`
raises. Resume is append+skip on `image_path`. Per-line print includes
`hyp_empty` and coverage so a real empty continuation is distinguishable
from a crash.

### 3. Re-baseline mean_coverage — **not computed here**

This laptop has **no** `checkpoints/demo/adapter_config.json` (SFT
adapter lived on Colab Drive). The diagnostic now exits 0 with
`mean_coverage: null` rather than writing a fake 0.0:

```
python src/models/demo/rlvr_train.py --sft-root checkpoints/demo --device cpu
# attempted: false, mean_coverage: null
```

The old Cell 10 figure **n=32, mean_coverage=0.0** is still
**untrustworthy**: it used images-only generate and silently converted
`generate()` crashes to `hyp=""`. It is **superseded** by a corrected
baseline once the fixed chat-template generate and exception handling
were used (Decision #88).

**Corrected SFT-greedy baseline (trustworthy, executed):**

From `docs/rlvr_sft_greedy_baseline.json` (committed for provenance):

- n = **32**
- n_empty_hyp = **0**  ✅ (real continuations; not crash→empty)
- mean_coverage = **0.309**
- n_coverage_lt_0.5 = **32** (32/32 below 0.5)
- chat_template = `sft_user_turn + add_generation_prompt=True`
- swallowed_exceptions = **false** ✅ (zero swallowed exceptions)

This is still **not** a retrained policy (no RL loop). It is the
baseline Phase 2b (λ=0 retrain) should be compared against.

### 4. Accuracy metric: **emitted-only** (both train-path R and gaming)

**Choice:** `emitted_only_accuracy` is the term inside `R`.

**Why:** Chapter 6’s omission game is “precision on what you said.”
`char_accuracy` is Levenshtein vs **full GT**, so deletions already
hurt; λ=0 would not recreate the unit-test cheat. Gaming analysis
(`ablation_omission_signal`) already used emitted-only; the diagnostic
used default `char_acc`. Those are now the same default
(`compute_reward(..., use_emitted_only_acc=True)`). Jsonl still stores
**both** `char_acc` and `emitted_acc`.

A future λ=0 retrain must keep `use_emitted_only_acc=True`. Do not
train on char_acc and then evaluate gaming with emitted-only.

Phase 2b still waits on a real SFT-greedy coverage from a machine that
has the adapter.


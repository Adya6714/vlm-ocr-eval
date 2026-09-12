# Tier 2 — RLVR coverage-term ablation

**Status: not trained. Do not read this as an omission result on a
model.**

BOOK.md Chapter 6 and Decision #11: only one cheap ablation — drop the
coverage term, retrain once, see whether the policy omits hard text.
A full reward-term sweep is out of scope.

## Dependency

RLVR needs a finished Stage 2b SFT adapter as the starting policy.
SFT **did not run** (no T4; inspect json missing; collate unwired).
Per the prompt: **RLVR training was not attempted**, including no
truncated one-epoch toy run.

## What exists (checkable without a GPU)

`src/models/demo/rlvr.py` defines

```text
R = char_acc + teds + tau − λ_coverage × (1 − coverage)
```

- `char_acc`: grapheme Levenshtein vs **GT length** (deletions count).
- `emitted_only_accuracy`: prefix-friendly channel used only in the
  diagnostic (this is how “say less” can look accurate).
- `coverage`: GT grapheme-cluster multiset overlap / |GT|.
- `teds` / `tau`: default 0.0 when no table tree / block permutation
  is supplied (inert, not invented).
- `λ_coverage = 0` is the ablation.

Unit tests (`tests/test_demo_rlvr_order.py`) confirm:

- Perfect copy → reward 1.0 with λ=1.
- Empty hyp vs nonempty GT → coverage 0, reward &lt; 0 with λ=1.
- With emitted-only accuracy and λ=0, a perfect easy prefix can beat a
  slightly noisy full reading; turning λ back to 1 penalizes the omit.

That is the **reward-shape** claim, not a trained-policy claim.

## What is needed to finish

1. Close Decision #3 and finish SFT (`docs/tier2_stage2b_demo.md`).
2. One RL train with λ=1 (coverage on), checkpoint.
3. One retrain from the **same SFT init** with λ=0.
4. Compare omission: coverage and length of hyp vs GT on a held slice
   of hard (rare-glyph) lines. Quantify; do not add extra ablations.

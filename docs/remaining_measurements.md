# Remaining measurements — run log

Ground truth for already-published numbers:
`docs/paper_defensibility_stats.md`. New numbers live in the files
named below, not in `paper/main.tex`.

## Step 0

See `docs/training_config.md` (0a + 0b).

**0a answer:** Table 5 agree/flip is **teacher-forced on the
full-memory greedy prefix**, not on ground truth and not a zero-memory
free-run. Quoted in that file from
`src/probes/probe_attention_ablation.py` lines 10–18 and 150–167.

## Step 1 — checkpoints

**Failed.** No `checkpoint_hindi_natural_seed{0,1,2}.pt` (and no
`tokenizer_hindi_natural.json`) in this checkout, `_local_archives/`
zips, or a Drive mount. Probe jsonl still points at
`/content/drive/MyDrive/vlm-ocr-eval/checkpoints/`. CPU wall-clock for
the first image was **not measured** (no weights).

Place the three `.pt` files and the matching tokenizer under a
gitignored `--output-root`, then:

```
PYTHONPATH=src python src/probes/probe_pos0_null.py --seed 0 --device cpu \
  --output-root checkpoints --data-root data \
  --out data/probe_results/probe_pos0_null_hindi_natural_seed0.jsonl
```

## Step 2 — E1

Code: `src/probes/probe_pos0_null.py`. **Not computed** until
`checkpoint_hindi_natural_seed{0,1,2}.pt` are available.
`src/analysis/analyze_pos0_null.py` writes a short report when jsonl exists.

## Step 3 — existing probes

`probe_gt_mismatch.py`, `probe_cross_attn_norms.py`,
`probe_gt_likelihood.py --extra-conditions noise scrambled` all need
the same checkpoints. **Not run.** Implementations already in
`src/probes/`.

## Step 4 — n-grams

**4a computed.** `docs/position_matched_ngrams.md`, producer
`src/analysis/position_matched_ngrams.py`. Same train/eval split as
stats §7.

**4b not computed.** jsonl has `step_p_gt` only, not the full softmax
or model argmax. Needs a checkpoint forward pass.

## Step 5 — Surya

Plumbing only: `docs/surya_positive_control.md`. **Diagnostic not
run.**

## Step 6 — ANOVA CIs

**Computed.** `docs/variance_share_bootstrap.md`, producer
`src/analysis/anova_share_bootstrap.py` (2,000 image-row resamples).

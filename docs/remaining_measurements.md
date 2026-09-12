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

**Failed (2026-09-12).** No `checkpoint_hindi_natural_seed{0,1,2}.pt`
and no `tokenizer_hindi_natural.json` in this checkout, `_local_archives/`
zips, or a Drive mount. Drive API: 403 insufficient scopes. See
`docs/tier0_checkpoint_status.md`. Probe jsonl still points at
`/content/drive/MyDrive/vlm-ocr-eval/checkpoints/`.

Place the three `.pt` files and the matching tokenizer under a
gitignored `--output-root`, then:

```
PYTHONPATH=src python src/probes/probe_pos0_null.py --seed 0 --device cpu \
  --output-root checkpoints --data-root data \
  --out data/probe_results/probe_pos0_null_hindi_natural_seed0.jsonl
```

## Step 2 — E1

**Not computed.** `docs/tier0a_pos0_null.md`.

## Step 3 — existing probes

**Not computed.** `docs/tier0b_gt_mismatch.md`,
`docs/tier0c_cross_attn_norms.md`, `docs/tier0d_noise_scrambled.md`.

## Step 4 — n-grams

**4a computed.** `docs/position_matched_ngrams.md`, producer
`src/analysis/position_matched_ngrams.py`. Same train/eval split as
stats §7.

**4b not computed.** jsonl has `step_p_gt` only, not the full softmax
or model argmax. Needs a checkpoint forward pass.

## Step 5 — Surya

**Diagnostic not run.** `docs/tier1_surya_control.md`.
surya-ocr 0.22 is llama.cpp (no encoder tensor / full softmax).
`vikp/surya_rec` weights were downloaded locally (~1.0 GB
`checkpoints/surya_v1_rec/model.safetensors`) but stock
`VisionEncoderDecoderModel.from_pretrained` **refuses** the checkpoint
(GQA 256 vs 1024, MoE expert keys, missing fc1/fc2). Custom
`LangVisionEncoderDecoderModel` + `SuryaProcessor` are not in this
environment. `ignore_mismatched_sizes` was **not** used.

## Step 6 — ANOVA CIs

**Computed.** `docs/variance_share_bootstrap.md`, producer
`src/analysis/anova_share_bootstrap.py` (2,000 image-row resamples).

## PaddleOCR full corpus (this session)

**Computed.** `docs/tier0e_paddleocr.md`. n=420 scored; exact 11;
Tier 1 among non-exact 4.2% (17/409). Tesseract/Surya Table 1 rows
unchanged.

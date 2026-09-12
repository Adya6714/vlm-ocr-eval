# Tier 0b — derangement teacher-forcing (`probe_gt_mismatch.py`)

**Status: not computed.** Blocked on Step 0
(`docs/tier0_checkpoint_status.md`). No mismatch jsonl. No Table 6
bucket means.

## What this would have measured

Producer: `src/probes/probe_gt_mismatch.py`. Scoring reuses
`probe_gt_likelihood.score_one` (teacher-forced `step_log_p_gt` on the
**paired** image’s GT string, pixels stay on `image_path`).

Same 60-image Hindi pool as Probe 5b / existing GT-likelihood jsonl.

Table 6 buckets (same as `paper_defensibility_stats.py`
`position_curve_block`, lines 769–776):

| Bucket | indices in `step_log_p_gt` |
|---|---|
| 0 | 0 |
| 1 | 1 |
| 2–9 | 2..9 |
| 10–19 | 10..19 |
| 20–39 | 20..39 |
| 40+ | ≥40 |

Per seed and pooled: mean and median of sequence-level `mean_log_p_gt`,
plus per-bucket mean log *p*(GT).

## Derangement vs 0a

See `docs/tier0a_pos0_null.md`. To share indices, run mismatch with
`--derange-seed $s` equal to `--seed $s`. Default `--derange-seed 0`
would **not** match pos0 seeds 1 and 2.

## Command (when weights exist)

```bash
for s in 0 1 2; do
  PYTHONPATH=src python src/probes/probe_gt_mismatch.py \
    --script hindi --condition natural --seed $s --derange-seed $s \
    --output-root checkpoints --data-root data --n-samples 100 \
    --device mps \
    --out data/probe_results/probe_gt_mismatch_hindi_natural_seed${s}.jsonl
done
```

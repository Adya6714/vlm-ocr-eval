# Tier 0d — noise and scrambled-patch teacher-forcing

**Status: not computed.** Blocked on Step 0
(`docs/tier0_checkpoint_status.md`). Existing
`data/probe_results/probe_gt_likelihood_hindi_natural_seed{0,1,2}.jsonl`
contain **real** and **blank** only. They were **not** extended, because
`--extra-conditions noise scrambled` needs a forward pass through the
checkpoints.

## What this would have measured

Producer: `src/probes/probe_gt_likelihood.py --extra-conditions noise scrambled`

- `noise`: `make_matched_noise` (Probe 3)
- `scrambled`: `make_scrambled_patches` (non-overlapping patch permute)

Same 60-image pool. Same Table 6 buckets as Tier 0b / `docs/paper_defensibility_stats.md`
Follow-Up 6 (cite that file for the existing real/blank numbers; do not
copy the table here as if it included noise/scrambled).

Resume key is `(condition, image_path)`, so extra conditions append
without rewriting real/blank rows.

## Command (when weights exist)

Write a **sibling** jsonl so the committed real/blank files stay
untouched until review:

```bash
for s in 0 1 2; do
  PYTHONPATH=src python src/probes/probe_gt_likelihood.py \
    --script hindi --condition natural --seed $s \
    --output-root checkpoints --data-root data --n-samples 100 \
    --device mps \
    --extra-conditions noise scrambled \
    --out data/probe_results/probe_gt_likelihood_extra_hindi_natural_seed${s}.jsonl
done
```

Using the original filename would mix new conditions into the paper’s
canonical jsonl; sibling path keeps that reversible.

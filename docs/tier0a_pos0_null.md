# Tier 0a — E1 position-0 null control

**Status: not computed.** Blocked on Step 0
(`docs/tier0_checkpoint_status.md`). No jsonl was written. No geometric
means, ranks, or entropies are reported here.

## What this would have measured

Producer: `src/probes/probe_pos0_null.py`
(`pos0_softmax` / `analyze_probs`, lines 70–128).

Same 60 Hindi Probe 5b images × 3 seeds = 180 (image, reference) pairs.
`--n-samples 100` caps at the pool size (60).

| Item | Computation in code |
|---|---|
| (a) geom. mean *p* of a uniform **non-argmax** symbol | 100 draws with replacement from V\{argmax} on the position-0 softmax; geom mean of those *p*; then geom mean across sequences (`analyze_pos0_null.geom_mean`) |
| (b) geom. mean *p*(GT) under image–reference **derangement** | `p_self[partner_gt_id]` at position 0; partner from `derangement_indices(n, np.random.default_rng(seed))` |
| (c) GT rank at position 0 | 1 = highest *p*; median, IQR, P(rank>100) |
| (d) Shannon entropy of position-0 softmax | `shannon_entropy` in nats, mean per seed |

Headline the paper wants is **(c)**, not 1/\|V\|.

## Derangement sharing with Tier 0b

`probe_pos0_null.py` uses `derangement_indices(n, default_rng(seed))`
(the **training seed**). `probe_gt_mismatch.py` uses
`--derange-seed` (CLI default **0**).

Those permutations **coincide only for seed 0** unless mismatch is run
with `--derange-seed {0,1,2}` matching the checkpoint seed. They were
not run, so no shared index file was written.

## Command (when weights exist)

```bash
for s in 0 1 2; do
  PYTHONPATH=src python src/probes/probe_pos0_null.py \
    --seed $s --device mps \
    --output-root checkpoints --data-root data --n-samples 100 \
    --out data/probe_results/probe_pos0_null_hindi_natural_seed${s}.jsonl
done
PYTHONPATH=src python src/analysis/analyze_pos0_null.py
```

`analyze_pos0_null.py` currently writes `docs/pos0_null_control.md`.
This file (`docs/tier0a_pos0_null.md`) is the task-facing report.

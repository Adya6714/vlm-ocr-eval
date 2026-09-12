# Tier 0c — cross-attention contribution norms

**Status: not computed.** Blocked on Step 0
(`docs/tier0_checkpoint_status.md`). No jsonl.

## What this would have measured

Producer: `src/probes/probe_cross_attn_norms.py`.

For each decoder layer ℓ, on a teacher-forced GT pass:

    r_ℓ = mean_t  ‖cross-attn output_t‖₂  /  ‖residual stream_t‖₂

Conditions: `real` vs `blank` (`CONDITIONS` in that file). Averaged over
positions, reported per seed and pooled. Requires the instrument
`TransformerDecoderLayer` hooks, not a closed OCR API.

## Command (when weights exist)

```bash
for s in 0 1 2; do
  PYTHONPATH=src python src/probes/probe_cross_attn_norms.py \
    --seed $s --output-root checkpoints --data-root data \
    --device mps \
    --out data/probe_results/cross_attn_norms_hindi_natural_seed${s}.jsonl
done
```

Confirm CLI flags in the script before running; this block matches
`--output-root` / `--seed` already required at line 349.

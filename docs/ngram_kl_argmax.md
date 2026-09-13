# n-gram KL and argmax agreement (Step 4b)

Target: convert Section 8's consistency claim (mid-sequence
teacher-forced log p(GT) ≈ 4-to-5-gram grapheme prior) into an
exclusivity claim: is the **full** decoder softmax the 5-gram
distribution, not merely matching it on the GT token?

Committed `probe_gt_likelihood_hindi_natural_seed*.jsonl` cannot
answer this: `score_one` in `src/probes/probe_gt_likelihood.py`
requests `return_full_probs=True` then **discards** the tensor
after entropy. No argmax, no KL(model ‖ 5-gram).

Producer (needs checkpoints): `src/probes/probe_ngram_kl.py`.
5-gram: add-α, α=0.01, same train/eval split as
`src/analysis/position_matched_ngrams.py` (manifest minus the 60
GT-likelihood real strings). KL uses generate.kl_divergence's
floor (1e-8) after scattering the 5-gram onto the tokenizer.

Buckets match Follow-Up 6 / Table 6: Position 0, 1, 2–9, 10–19,
20–39, 40+ on teacher-forced step index (content + trailing EOS).

**Status: not computed.** This laptop has no Hindi instrument
checkpoints (`docs/tier0_checkpoint_status.md`). Compact jsonl
was not written. Missing:
- `data/probe_results/probe_ngram_kl_hindi_natural_seed0.jsonl`
- `data/probe_results/probe_ngram_kl_hindi_natural_seed1.jsonl`
- `data/probe_results/probe_ngram_kl_hindi_natural_seed2.jsonl`

Command on T4 or CPU once `--output-root` has
`checkpoint_hindi_natural_seed{0,1,2}.pt` and
`tokenizer_hindi_natural.json`:

```bash
for s in 0 1 2; do
  PYTHONPATH=src python src/probes/probe_ngram_kl.py \
    --script hindi --condition natural --seed $s \
    --output-root checkpoints --data-root data --n-samples 100 \
    --device cuda \
    --out data/probe_results/probe_ngram_kl_hindi_natural_seed${s}.jsonl
done
PYTHONPATH=src python src/analysis/ngram_kl_argmax.py
```

Do not backfill KL from `step_p_gt`. That is one coordinate of
the softmax, not the distribution.

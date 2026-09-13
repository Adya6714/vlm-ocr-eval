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

### Seed 0

| Bucket | n steps | mean KL(model ‖ 5-gram) | argmax agreement |
|---|---:|---:|---:|
| Position 0 | 60 | 4.3638 | 0.1333 |
| Position 1 | 60 | 1.3244 | 0.7167 |
| Positions 2–9 | 480 | 0.3017 | 0.9375 |
| Positions 10–19 | 600 | 0.2481 | 0.9183 |
| Positions 20–39 | 879 | 0.2635 | 0.9147 |
| Positions 40+ | 378 | 1.7701 | 0.6376 |

**Partial.** Missing or empty:
- `data/probe_results/probe_ngram_kl_hindi_natural_seed1.jsonl`
- `data/probe_results/probe_ngram_kl_hindi_natural_seed2.jsonl`

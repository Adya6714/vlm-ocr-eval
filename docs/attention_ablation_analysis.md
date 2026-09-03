# Attention ablation — does confidence depend on the image?

**Generated:** 2026-09-03

**Inputs:**
- `data/probe_results/attention_ablation_hindi_natural_seed0.jsonl`
- `data/probe_results/attention_ablation_hindi_natural_seed1.jsonl`
- `data/probe_results/attention_ablation_hindi_natural_seed2.jsonl`

**Sample:** Hindi/natural Tier C real images; `Random(0)` draw (pool size 60),
60 images per seed × 3 seeds = **180** images.

## Method (mechanism probe; inference only)
For each image, we run the instrument’s greedy decode twice:
1) with **full encoder memory**, and
2) with encoder memory replaced by **all zeros before** `memory_projection`.

For per-step distribution comparison under a clean “same prefix” condition,
we also score the zero-memory decoder **teacher-forced** along the full-memory
token sequence and compute per-step:
- **KL(full || zero)** (primary; KL(zero || full) is also stored),
- **Top-1 agreement** (argmax token equality), and
- **prior sufficiency** = `sum_i min(p_full[i], p_zero[i])` = `1 − TV(p_full,p_zero)`.

## Pooled 3-seed results (180 images)
- `mean_confidence_full` = **0.9861**
- `mean_confidence_zero` = **0.9891**
- `delta` (full − zero) = **-0.0030**
- `top1_agreement_rate` = **0.8794**
- `mean_prior_sufficiency` = **0.8827**
- `mean KL(full||zero)` = **1.075**

### Plain-language finding
Confidence is **~unaffected** by deleting all encoder information: the decoder’s
mean confidence barely changes (delta ≈ -0.003).

But the distributions are not identical: prior sufficiency 0.8827 implies
`1 − prior_sufficiency` ≈ **0.1173 (~12%)** of the probability mass lies outside
the prior-only overlap. In other words, **content and confidence are dissociated**:
confidence looks prior-dominated, while only about **~12%** of token choice mass
still depends on having the image.

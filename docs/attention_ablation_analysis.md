# Attention ablation — encoder memory vs decoder confidence

**Generated:** 2026-09-03 · **rewritten:** 2026-09-04 (seed-honest pass)

**Inputs:**
- `data/probe_results/attention_ablation_hindi_natural_seed0.jsonl`
- `data/probe_results/attention_ablation_hindi_natural_seed1.jsonl`
- `data/probe_results/attention_ablation_hindi_natural_seed2.jsonl`

**Sample:** Hindi/natural Tier C real images; `Random(0)` draw (pool size 60),
60 images per seed × 3 seeds = **180** images. All numbers below recomputed
from those jsonl files.

## Method (mechanism probe; inference only)
For each image, greedy decode twice:
1) with **full encoder memory**, and
2) with encoder memory replaced by **all zeros before** `memory_projection`.

Per-step distribution comparison uses the zero-memory decoder **teacher-forced**
along the full-memory token sequence:
- **KL(full ‖ zero)** (primary; KL(zero ‖ full) also stored),
- **Top-1 agreement** (argmax token equality),
- **prior sufficiency** = `sum_i min(p_full[i], p_zero[i])` = `1 − TV(p_full, p_zero)`.

Confidence here is mean max-softmax over self-generated tokens
(`mean_confidence_full` / `mean_confidence_zero`).

---

## Per-seed results (read this first)

| seed | n | mean_conf_full | mean_conf_zero | Δ (full − zero) | mean KL(full‖zero) | top-1 agreement | mean prior sufficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 0.9755371535 | 0.9988899241 | **−0.0233527706** | 1.6386843913 | 0.8567750668 | 0.8603082788 |
| 1 | 60 | 0.9923365538 | 0.9888020617 | **+0.0035344921** | 0.4397526702 | 0.9504229482 | 0.9515855914 |
| 2 | 60 | 0.9905533814 | 0.9796793436 | **+0.0108740378** | 1.1466236148 | 0.8310073835 | 0.8362425902 |

- Per-seed confidence deltas: **−0.0233527706**, **+0.0035344921**, **+0.0108740378**
- Span of those deltas: **0.0342268084**
- Signs disagree across seeds (seed 0 negative; seeds 1–2 positive).

### What the pooled confidence delta is not
Pooled over 180 images:

- `mean_confidence_full` = **0.9861423629**
- `mean_confidence_zero` = **0.9891237765**
- `delta` (full − zero) = **−0.0029814136**

That pooled Δ ≈ −0.003 is **cancellation across seeds with opposite signs**,
not a stable small effect that each seed reproduces. Do not cite the pooled
confidence delta as evidence that “confidence is unaffected by the encoder.”

---

## Primary finding: the distribution moves; the max does not move consistently

Pooled distribution metrics (180 images; these are stable in direction across
seeds even when confidence Δ flips sign):

- `mean KL(full ‖ zero)` = **1.0750202254** nats
- `top1_agreement_rate` = **0.8794017995**
- `mean_prior_sufficiency` = **0.8827121535**
  → `1 − prior_sufficiency` ≈ **0.1172878465** (~11.7% of probability mass
  outside the prior-only overlap)

**Claim (precise):** under encoder-memory ablation, the output **distribution**
changes substantially (mean KL ≈ 1.075 nats; top-1 agreement ≈ 0.879). The
**maximum** of that distribution (mean max-softmax confidence) does **not**
move consistently in either direction across seeds. Content and confidence are
dissociated in that sense — not because confidence is proven invariant to the
encoder, but because a large distributional shift coexists with an unstable,
near-zero mean confidence delta.

## Not in these jsonl files

Per-layer cross-attention contribution norms
(`‖cross-attn output‖ / ‖residual stream‖`) are **not** recorded here.
That measurement is `src/probes/probe_cross_attn_norms.py` (Colab;
DECISIONS.md #63) and is required before claiming a residual-mass
"modality bypass" percentage.

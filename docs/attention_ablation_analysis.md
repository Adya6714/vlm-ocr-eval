# Attention ablation — does confidence depend on the image?

**Status:** methods + code ready; **numbers pending** Colab inference
on `checkpoint_hindi_natural_seed{0,1,2}.pt` (not present on the
laptop checkout that authored this stub).

**Code:** `src/probes/probe_attention_ablation.py`,
`src/analysis/analyze_attention_ablation.py`,
`src/models/instrument/generate.py` (`zero_encoder_memory`,
`force_next_ids`, `return_full_probs`).  
**Decision:** DECISIONS.md #56.  
**Expected outputs:**
`data/probe_results/attention_ablation_hindi_natural_seed{0,1,2}.jsonl`

---

## Method (locked before looking at results)

1. **Sample.** Same Hindi Tier C draw as Probe 5b
   (`random.Random(0)`, `--n-samples 100`, capped by pool size — 60
   images in the current `data/raw/hindi/ground_truth.jsonl`).
2. **Full memory.** Ordinary `generate()` — real encoder features,
   projected, cross-attended.
3. **Zero memory (confidence).** `generate(..., zero_encoder_memory=True)`
   — encoder output replaced with zeros **before** `memory_projection`;
   independent greedy decode. Headline = `mean_confidence_full` vs
   `mean_confidence_zero`.
4. **Zero memory (distributions).** Teacher-force the zero-memory
   decoder along the full-memory token sequence. Per step:
   - **KL(full || zero)** (primary; reverse also stored)
   - **Top-1 agreement** (argmax equal?)
   - **Prior sufficiency** = `sum_i min(p_full[i], p_zero[i])` = `1 − TV`
5. **Stats.** Per-seed means; across-seed mean±SD; paired cluster
   bootstrap of images (n_boot = 10_000), same repair family as
   Probe 5b / `docs/statistical_repair.md`.

---

## Colab run (after Drive checkpoints are mounted)

```bash
for s in 0 1 2; do
  PYTHONPATH=src/probes:src/models/instrument python3 \
    src/probes/probe_attention_ablation.py \
    --script hindi --condition natural --seed $s \
    --output-root "$CKPT_ROOT" \
    --data-root data \
    --n-samples 100 \
    --device cuda \
    --out data/probe_results/attention_ablation_hindi_natural_seed${s}.jsonl
done

PYTHONPATH=src/analysis python3 \
  src/analysis/analyze_attention_ablation.py \
  --probe-dir data/probe_results \
  --out docs/attention_ablation_analysis.md
```

Commit and push the three jsonl files immediately after generation
(AGENTS.md probe_results rule).

---

## Results

*Waiting on Colab.* Re-run `analyze_attention_ablation.py` to replace
this section with the across-seed table (confidence Δ, KL, top-1
agreement, prior sufficiency) and the plain-language finding.

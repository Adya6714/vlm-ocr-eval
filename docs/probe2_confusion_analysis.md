# Probe 2 — confusion structure (GT-aligned)

**Status:** code + unit tests ready; **numbers pending** Colab
inference on `checkpoint_hindi_natural_seed{0,1,2}.pt` (not on the
laptop checkout).

**Code:** `src/probes/probe2_confusion_graph.py`,
`src/analysis/analyze_probe2.py`  
**Decisions:** #47 (script-scoped checkpoint names — verified),
#57 (GT-aligned + p(true))  
**Expected outputs:**
`data/probe_results/probe2_hindi_natural_seed{0,1,2}.jsonl`

---

## Method (locked)

1. **Sample.** Same Hindi Tier C draw as Probe 5b (`random.Random(0)`,
   `--n-samples 100`, pool currently 60).
2. **Generate** with `return_full_probs=True` (top-5 alone cannot
   recover p(true) when rank > 5).
3. **Align** pred vs GT grapheme clusters (Needleman–Wunsch).
4. **Per substitution:** true, predicted, top-5, p(true), rank(true).
5. **Report:** top-15 pairs; mean p(true) / mean rank on misreads;
   qualitative tags on the printed pairs (`same-base-matra-diff`,
   `adjacent-codepoint`, `dissimilar`, …).

Checkpoint path printed at startup:
`{output_root}/checkpoint_hindi_natural_seed{N}.pt` — legacy
`checkpoint_natural_seed{N}.pt` is **not** accepted.

---

## Colab run

```bash
for s in 0 1 2; do
  PYTHONPATH=src/probes:src/models/instrument python3 \
    src/probes/probe2_confusion_graph.py \
    --script hindi --condition natural --seed $s \
    --output-root "$CKPT_ROOT" \
    --data-root data \
    --n-samples 100 \
    --device cuda \
    --out data/probe_results/probe2_hindi_natural_seed${s}.jsonl
done

PYTHONPATH=src/analysis:src/probes python3 \
  src/analysis/analyze_probe2.py \
  --probe-dir data/probe_results \
  --out docs/probe2_confusion_analysis.md
```

Commit and push the three jsonl files immediately (AGENTS.md).

---

## Results

*Waiting on Colab.* Re-run `analyze_probe2.py` to fill the across-seed
mean p(true) table, top-15 pairs, and the plain-language finding
(close-but-wrong vs completely off).

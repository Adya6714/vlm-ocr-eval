# Position-matched n-gram references (Step 4a)

**Source:** `data/manifests/hindi_natural.jsonl` excluding the 60 Probe
GT-likelihood real strings (same split as
`docs/paper_defensibility_stats.md` §7).
**Train lines:** 2491. **Eval strings:** 60.
**V_est:** 367 (unique train graphemes + 2). **add-α:** 0.01.

Instrument bucket means are recomputed from
`data/probe_results/probe_gt_likelihood_hindi_natural_seed{0,1,2}.jsonl`
(same pooling as Follow-Up 6).

Step 4b (mean KL(model ‖ 5-gram) and argmax agreement) is **not computed**:
committed `probe_gt_likelihood_*.jsonl` stores `step_p_gt` / entropy only,
not the full softmax or the model argmax.

| Bucket | n tokens (eval) | unigram | bigram | trigram | 4-gram | 5-gram | instrument real (FU6) | instrument blank (FU6) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Position 0 | 60 | -5.9189 | -4.7307 | -4.7307 | -4.7307 | -4.7307 | -24.5384 | -23.4835 |
| Position 1 | 60 | -5.0757 | -1.3771 | -1.1380 | -1.1380 | -1.1380 | -8.1163 | -8.6301 |
| Positions 2–9 | 480 | -4.1113 | -2.2516 | -0.8813 | -0.3807 | -0.3055 | -0.4784 | -0.4092 |
| Positions 10–19 | 598 | -4.1822 | -2.3350 | -1.0602 | -0.4471 | -0.2633 | -0.1478 | -0.1578 |
| Positions 20–39 | 851 | -4.1270 | -2.2394 | -0.9814 | -0.4276 | -0.2021 | -0.2726 | -0.2763 |
| Positions 40+ | 348 | -4.1681 | -2.2776 | -0.9867 | -0.4001 | -0.1808 | -6.1153 | -6.2835 |

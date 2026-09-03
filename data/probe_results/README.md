# probe_results/

Committed outputs from `src/probes/`. One family per probe. Re-running a
probe must resume (skip existing ids) and must not overwrite blindly.

| Prefix | Seeds | What each record is |
|---|---|---|
| `probe2_hindi_natural_seedN.jsonl` | 0–2 | Single JSON object with `edges` (not line-delimited). |
| `probe3_hindi_{natural,flattened,inverted}_seedN.jsonl` | 0–2 | One line-crop per line: real/blank/noise confidence. |
| `probe3_curve_hindi_natural_seedN.json` | 0–2 | Training-step snapshots (not jsonl). |
| `probe5_hindi_{natural,flattened,inverted}_seedN.jsonl` | 0–2 | Calibration records + buckets. |
| `probe5b_hindi_natural_seedN.jsonl` | 0–2 | hindi/santhali/kashmiri/blank; 720 rows total. |
| `attention_ablation_hindi_natural_seedN.jsonl` | 0–2 | Full vs zero-memory confidence + KL scalars. |
| `probe6_synthetic_real_hindi_seedN.jsonl` | 0–2 | real_plain / real_degraded / blank on Tier C. |
| `sarvam_transfer_probe.jsonl` | — | 35 Extract pages (10+10+10+5 blank). |
| `summary_hindi.json` | — | Pooled Probe 3/5 aggregates from the Hindi zip. |

Trace a headline: [`docs/RESULTS.md`](../../docs/RESULTS.md).

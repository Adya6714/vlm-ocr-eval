# What ran where

Heavy training and OCR batches run on Colab (free T4). This laptop
holds the extracted numbers, not the checkpoints.

**Canonical numbers:** [`docs/RESULTS.md`](./docs/RESULTS.md)  
**Committed jsonl:** `data/probe_results/`  
**Local zip / checkpoint blobs:** `_local_archives/` (gitignored)

| Job | Machine | Lands in-repo as |
|---|---|---|
| Hindi instrument 3×3 + Probes 3/5 | Colab | `data/probe_results/probe3_*.jsonl`, `probe5_*.jsonl` |
| Probe 3b curve, Probe 5b | Colab | `probe3_curve_*.json`, `probe5b_*.jsonl` |
| Attention ablation, Probe 2, Probe 6 | Colab | `attention_ablation_*.jsonl`, `probe2_*.jsonl`, `probe6_*.jsonl` |
| GT-likelihood (teacher-forced) | Colab | `probe_gt_likelihood_hindi_natural_seed{0,1,2}.jsonl` |
| Stage 5a Extract (35 pages) | this laptop | `sarvam_transfer_probe.jsonl` + `data/cache/sarvam/` |
| Stage 0 baselines | Colab / local | `data/predictions/` (gitignored) |
| Bengali 3×3 checkpoints | Colab | `_local_archives/Bengali Experiment Final.zip` only — **no Bengali probe jsonl** |
| Manifest export (100 pages/mode) | Colab | `data/manifests/{hindi,bengali}_*.jsonl` |

Do not drop a new zip on the repo root. Extract into the IMPLEMENTATION.md
path for that stage, then move the zip to `_local_archives/`.

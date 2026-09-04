# Where the numbers live

This is the index for anyone who wants to trace a headline figure back
to the file that produced it. Do not recompute from chat history.

**Raw probe outputs (committed):** `data/probe_results/`  
**Claim-facing write-ups:** this `docs/` folder  
**Code that wrote the jsonl:** `src/probes/`  
**Code that wrote the write-up:** `src/analysis/`  
**Stage 0 engine outputs (local, gitignored):** `data/predictions/`  
**Training manifests (committed):** `data/manifests/`  
**Heavy Colab zips / checkpoints (local only):** `_local_archives/`

Status and acceptance criteria: [`IMPLEMENTATION.md`](../IMPLEMENTATION.md).  
Design choices: [`DECISIONS.md`](../DECISIONS.md).

---

## Headline → evidence

| Claim | Number (in-tree) | Output file(s) | Analysis | Producer |
|---|---|---|---|---|
| Stage 0 Tier 1 among Tesseract non-exact | 20.4% (provisional) | `data/predictions/error_taxonomy.csv` (local) | [`adjudication_analysis.md`](adjudication_analysis.md), [`tier2_validation.md`](tier2_validation.md) | `src/eval/error_taxonomy.py` |
| Stage 1 glyph-frequency TV | natural 0 / flat ≈0.047 / inv ≈0.005 | manifests under `data/manifests/hindi_*.jsonl` | IMPLEMENTATION Stage 1 | `src/renderer/glyph_frequency.py` |
| Probe 1 exposure β | **withheld** (flat/inv line acc ~0%) | Probe 5 jsonl + manifests | [`probe1_fixed_effects.md`](probe1_fixed_effects.md) | `src/analysis/probe1_fixed_effects.py` |
| Probe 2 confusion | EOS/space weight 0.0891; mixed top pairs | `probe2_hindi_natural_seed{0,1,2}.jsonl` | [`probe2_confusion_analysis.md`](probe2_confusion_analysis.md) | `src/probes/probe2_confusion_graph.py` |
| Probe 3 real/blank/noise (natural, 3-seed) | 0.9940 / 0.9899 / 0.9904 | `probe3_hindi_{natural,flattened,inverted}_seed{0,1,2}.jsonl` | [`results_analysis.md`](results_analysis.md) | `src/probes/probe3_blank_control.py` |
| Probe 3b training curve | gap ≈ 0; loss 3.210 → 0.182 | `probe3_curve_hindi_natural_seed{0,1,2}.json` | [`probe3_curve_analysis.md`](probe3_curve_analysis.md) | `src/probes/probe3_training_curve.py` |
| Probe 5 calibration (natural) | ~99% conf, 14–24% acc, ECE ≈ 0.81 | `probe5_hindi_*_seed{0,1,2}.jsonl` | [`results_analysis.md`](results_analysis.md) | `src/probes/probe5_calibration.py` |
| Probe 5b zero-shot floor | 720 records; cond. range 0.0037; 360/360 substitution | `probe5b_hindi_natural_seed{0,1,2}.jsonl` | [`probe5b_analysis.md`](probe5b_analysis.md), [`statistical_repair.md`](statistical_repair.md) | `src/probes/probe5b_zeroshot_floor.py` |
| Attention ablation | full 0.9861 vs zero 0.9891; Δ −0.0030; prior suff. 0.8827 | `attention_ablation_hindi_natural_seed{0,1,2}.jsonl` | [`attention_ablation_analysis.md`](attention_ablation_analysis.md) | `src/probes/probe_attention_ablation.py` |
| GT-likelihood (teacher-forced) | log p(GT) real −1.783 / blank −1.751; first-token ~1e-10–1e-12 both | `probe_gt_likelihood_hindi_natural_seed{0,1,2}.jsonl` | [`gt_likelihood_analysis.md`](gt_likelihood_analysis.md) | `src/probes/probe_gt_likelihood.py` |
| Probe 6 Tier C | conf 0.9861 / 0.9768 / 0.9799; acc 0.0; 0 leakage | `probe6_synthetic_real_hindi_seed{0,1,2}.jsonl` | [`probe6_synthetic_real_analysis.md`](probe6_synthetic_real_analysis.md) | `src/probes/probe6_synthetic_real_gap.py` |
| Stage 5a transfer | published gap 39.98 pp vs conf Δ 0.0027 | `sarvam_transfer_probe.jsonl` | [`sarvam_transfer_analysis.md`](sarvam_transfer_analysis.md) | `src/probes/sarvam_transfer_probe.py` |

Kashmiri Bonferroni “significance” on Probe 5b seed 0 is **retracted** (`statistical_repair.md`, DECISIONS.md #53).

---

## Pipeline (what calls what)

```
data/raw/{hindi,bengali,santhali,kashmiri}/     GlotOCR images + GT   (local)
        │
        ├─► src/eval/run_baselines.py  →  data/predictions/{engine}/
        │         └─► error_taxonomy.py → error_taxonomy.csv
        │
        └─► src/renderer + export_manifest_scaled.py
                  └─► data/manifests/{script}_{mode}.jsonl
                            └─► src/models/instrument/train.py   (Colab)
                                      └─► checkpoints/*.pt        (local / Drive)
                                                └─► src/probes/*.py
                                                          └─► data/probe_results/*.jsonl
                                                                    └─► src/analysis/*.py
                                                                              └─► docs/*_analysis.md
```

---

## What is *not* in git (and why)

| Path | Why excluded |
|---|---|
| `data/raw/`, `data/cache/` | Images and renderer caches. Rebuild or fetch. |
| `data/predictions/` | Engine jsonl; regenerable via `run_baselines.py`. |
| `checkpoints/` | ~200 MB each. Colab Drive / `_local_archives/Bengali Experiment Final.zip`. |
| `_local_archives/*.zip` | Colab export blobs after jsonl was extracted into `data/probe_results/`. |
| TrOCR run | Excluded on purpose: in-domain conf 0.42, wrong regime. |

If a number is not in the table above, it is not a headline claim.

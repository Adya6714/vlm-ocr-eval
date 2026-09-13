# Tier 2 / Stage 2b — T4 LoRA benchmark and SFT (Colab)

**Selected:** `ds4sd/SmolDocling-256M-preview`

**Why:** lowest dummy-LoRA peak among four candidates that all fit a
T4 with ≥2 GB headroom (1.63 GB peak, 14.37 GB left). Closed as
Decision #84 (do not rewrite #3).

## Dummy LoRA peaks (one Colab T4 pass, batch_size=1, 3 steps)

| HF id | peak | dummy loss | notes |
|---|---|---|---|
| `ds4sd/SmolDocling-256M-preview` | 1.63 GB | 10.7→10.2 | idefics3 |
| `ibm-granite/granite-docling-258M` | 1.84 GB | nan | VRAM only; not selected |
| `lightonai/LightOnOCR-1B-1025` | 5.67 GB | 12.6→11.3 | mistral3; `image_token_id=151655`, 196 placeholders |
| `lightonai/LightOnOCR-2-1B-base` | 5.67 GB | 10.9→10.0 | same family |

SFT `target_modules` are the inspect leaf names from that same Cell 8
pass (including `lm_head`, `embed_tokens`, vision embeddings, `fc1`/`fc2`).
The VRAM dummy used the identical list. Not a config mismatch.

## SFT

100 steps on `data/manifests/hindi_natural.jsonl` (2538 lines), CUDA,
`checkpoints/demo/` with `adapter_config.json` and step_20…step_100.
Step 1 loss=22.97, step 100 loss=0.51. Line crops came from Drive
`data/cache/line_crops` (gitignored). This is a smoke-test LoRA, not a
finished OCR system.

## RLVR

Not a coverage ablation. See `docs/tier2_rlvr_ablation.md`.

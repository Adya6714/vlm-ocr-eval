## Demo SFT adapter: where it goes

The demo SFT run produces a PEFT LoRA adapter for the base model:
`ds4sd/SmolDocling-256M-preview` (Decision #84).

- **Correct runtime location (not committed)**: `checkpoints/demo/`
  - This directory is gitignored by design (`checkpoints/` is heavy).
  - It should contain `adapter_config.json` plus a weights file such as
    `adapter_model.safetensors`, and optional step snapshots under
    `step_20/…step_100/`.

If you receive the adapter as a zip or a loose folder named
`Demo Adapter/`, copy its contents into `checkpoints/demo/` and do not
commit the weights.

### What we commit vs what we don’t

- **Committed (small, reproducible)**:
  - `docs/tier2_stage2b_demo.md` (human-readable run summary)
  - `docs/demo_sft_meta.json` and `docs/demo_sft_state.json` (machine-readable run metadata)
- **Not committed (large, regenerable / transferable)**:
  - `checkpoints/demo/**` (all adapter weights + snapshots)

### Integrity note (this adapter blob)

- **Weights file**: `checkpoints/demo/adapter_model.safetensors`
- **Size**: 151,197,304 bytes (~151 MB)
- **SHA-256**: `9ef910c50ae28e679ab22bdbd26553b630ec018c5f3af9788d4f33dacd7fa9d3`

## Minimal “attach this to another AI agent” bundle

If another agent needs to run analysis or inference with this adapter, attach:

- **Adapter folder (required)**: a zip of `checkpoints/demo/` (includes `adapter_config.json` and weights)
- **Base model id (required)**: `ds4sd/SmolDocling-256M-preview`
- **Prompt/collate contract (required)**:
  - `src/models/demo/sft.py` (especially `SFT_USER_INSTRUCTION`, `sft_user_turn()`,
    `encode_for_generate()`, `decode_continuation()`)
- **Training data reference (recommended)**:
  - `data/manifests/hindi_natural.jsonl`
  - If the agent will *train* or *rerun SFT*, they also need the line crops
    `data/cache/line_crops/hindi/` (gitignored) or they must regenerate them in Colab.
- **Run metadata (recommended)**:
  - `docs/tier2_stage2b_demo.md`
  - `docs/demo_sft_meta.json`
  - `docs/demo_sft_state.json`

## What analysis to ask the next agent to do

The adapter alone is not an “evaluation.” Suggested next steps (GPU/Colab):

- Run a small **SFT sanity eval** on a held-out slice of `hindi_natural.jsonl`:
  - Load base model + adapter
  - Sample ~50 lines, decode with the exact chat template path used in SFT
  - Report Tier-1-aware exactness / CER using the repo scorer (Stage 0 logic)
- Optionally run **RLVR** only after confirming the SFT decoding path is correct
  (the repo deliberately refuses “images-only” generation for this demo).


# Tier 2 / Stage 2b — demo LoRA (honest partial)

**Host:** macOS, `torch.cuda.is_available() == False`, MPS available.
**This is not a T4 run.** Decision #3 stays **open**. Nothing below is a
peak-VRAM number.

## Step 1 — Decision #3 (the blocking measurement)

`src/models/demo/benchmark_base_models.py` is wired for:

| HF id | Role |
|---|---|
| `ds4sd/SmolDocling-256M-preview` | Decision #3 candidate |
| `ibm-granite/granite-docling-258M` | successor, same family |
| `lightonai/LightOnOCR-1B-1025` | Decision #3 candidate |
| `lightonai/LightOnOCR-2-1B` | successor |
| `lightonai/LightOnOCR-2-1B-base` | fine-tune variant if LightOn wins later |

**What ran**

- CUDA dummy-train path **refused** on this host (`CUDA required for T4
  VRAM numbers`). MPS was not substituted.
- `--inspect` for SmolDocling was **started** (transformers 5.15.1,
  `AutoModelForImageTextToText`). A Hub download reached **402 MB**
  incomplete (`models--ds4sd--SmolDocling-256M-preview` blob) and then
  stalled at 0% CPU. The process was killed. **No
  `docs/demo_lora_inspect.json` was written.** Leaf `target_modules`
  were **not** obtained. They are still not to be guessed.

**Call (explicit):** do **not** close Decision #3. No model is known to
fit a T4 under LoRA. No headroom number exists for later reading-order /
table modules. `DEFAULT_MODEL_ID = ds4sd/SmolDocling-256M-preview` is a
**development default for wiring**, not a hardware decision
(DECISIONS.md #79).

**To finish on a free T4**

```text
python src/models/demo/benchmark_base_models.py --model-id ds4sd/SmolDocling-256M-preview --inspect
python src/models/demo/benchmark_base_models.py --model-id ibm-granite/granite-docling-258M --inspect
# then LoRA dummy steps with the printed leaf names, fp16, no FA2
python src/models/demo/benchmark_base_models.py --model-id ... --target-modules <from inspect>
# same for LightOnOCR-1B-1025 and LightOnOCR-2-1B-base if time
```

Record peak GB vs 16 GB T4; leave ≥2 GB for a later detector if both
would be resident. Then append a Decision #3 close (do not rewrite #3).

## Step 2 — LoRA stack (code, not a trained adapter)

| File | What it is | Run? |
|---|---|---|
| `src/models/demo/base_model.py` | loader + inspect helper | load **not** completed |
| `src/models/demo/lora_config.py` | PEFT config; **refuses** without inspect json | no SFT |
| `src/models/demo/layout_module.py` | **oracle** boxes from PageGT lines, not a detector | unit test only |
| `src/models/demo/reading_order_module.py` | pairwise count-and-sort (geometry) | unit test only |
| `src/models/demo/sft.py` | dataset + `--run` | `--run` **refused** (no CUDA); processor collate **not wired** |

`peft` is in `requirements.txt`; this session’s default interpreter
did not need it because training never started.

### Design choice — training corpus (flagged, not silent)

**Chosen (when a T4 exists):** `data/manifests/hindi_natural.jsonl`
(2538 line crops, exact GT already in-repo, Colab `--data-root`
portable). Optional later: concatenate `bengali_natural.jsonl` (1425
lines) for a two-script demo.

**Why this and not GlotOCR page images:** the demo is a practical
system demo (BOOK.md Ch. 4), not Probe 1. Line crops with exact GT are
the same contract as the instrument trainer, already rendered, and
cheap enough for a T4 LoRA loop. Full pages would be more
“production-like” (layout + reading order in one sequence) but the
renderer still does not paint table/form cells, and GlotOCR pages have
no layout tags.

**Why this is a design choice:** SFT on line crops will **not** teach
block order or table binding. Those stay separate modules (pairwise
orderer + future detector). A reviewer should not read a line-crop LoRA
as “the demo reads two-column pages.”

## What is needed to finish

1. T4 inspect + LoRA peak VRAM → close #3.
2. Write `docs/demo_lora_inspect.json` from that inspect (real leaf names).
3. Wire processor collate in `sft.py` (chat template is model-specific;
   cannot be invented before inspect).
4. Train with fp16 + gradient checkpointing, checkpoint under
   `checkpoints/demo/`, resume by default.
5. Do **not** start RLVR until that adapter exists
   (`docs/tier2_rlvr_ablation.md`).

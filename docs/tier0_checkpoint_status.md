# Step 0 — Hindi instrument checkpoints

**Status: blocked.** No `checkpoint_hindi_natural_seed{0,1,2}.pt` and no
`tokenizer_hindi_natural.json` were present on this machine. Nothing
below this line was downloaded, estimated, or substituted.

## What was searched (this session, 2026-09-12)

| Location | Result |
|---|---|
| `checkpoints/` (gitignored `--output-root` expected by probes) | directory missing |
| Spotlight `checkpoint_hindi_natural_seed0.pt` | no hits |
| `/Users/adya/Downloads`, `Desktop`, `Documents`, repo tree | no Hindi `.pt` |
| Google Drive Desktop / `~/Library/CloudStorage` | not mounted |
| DriveFS | not present |
| `_local_archives/Bengali Experiment Final.zip` | Bengali-only weights named `checkpoint_{natural,flattened,inverted}_seed{N}.pt` — **not used** |
| `_local_archives/Hindi Probe Results Final.zip` | jsonl only, no weights |
| `gcloud auth print-access-token` → Drive v3 files.list | **403** `ACCESS_TOKEN_SCOPE_INSUFFICIENT` |
| `gcloud auth login --enable-gdrive-access` | started; no interactive completion in this session |

Committed probe jsonl still records Colab paths, e.g.
`/content/drive/MyDrive/vlm-ocr-eval/checkpoints/checkpoint_hindi_natural_seed0.pt`
(`data/probe_results/probe6_synthetic_real_hindi_seed0.jsonl`).

## Expected layout once files exist

```
checkpoints/checkpoint_hindi_natural_seed0.pt
checkpoints/checkpoint_hindi_natural_seed1.pt
checkpoints/checkpoint_hindi_natural_seed2.pt
checkpoints/tokenizer_hindi_natural.json
```

`--output-root checkpoints` matches `src/models/instrument/train.py`
`checkpoint_path` / `tokenizer_path` (DECISIONS.md #47). `checkpoints/`
is gitignored.

**File sizes:** not reported — files were not obtained.

This blocks Tier 0a–0d (all require those three checkpoints). Tier 0e
(PaddleOCR) and the Surya **plumbing** investigation do not.

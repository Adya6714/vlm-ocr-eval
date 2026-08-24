# Colab Run Log

## export_manifest_scaled.py — full run (hindi + bengali, 100 pages/mode)

- Status: in progress
- Params: --pages-per-mode 100, both scripts
- Output: data/manifests/{hindi,bengali}\_{natural,flattened,inverted}.jsonl

## run_baselines.py — full run (all 4 languages)

- Status: completed
- Output: zip folder, exact repo path TBD — see audit below
- Note: `--help` was also run separately just to check usage, produced no results of its own

## export_manifest_scaled.py — dry run (hindi, 3 pages/mode)

- Status: completed, ran in local/Cursor sandbox, not Colab
- Purpose: verify resumability before committing to the full Colab job
- Output: data/manifests/hindi\_{natural,flattened,inverted}.jsonl (small test versions, since overwritten... check with Cursor)

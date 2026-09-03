# data/

| Path | In git? | What it is |
|---|---|---|
| `probe_results/` | yes | Committed probe jsonl — the numbers. See `probe_results/README.md`. |
| `manifests/` | yes | Line-crop training manifests (`{script}_{mode}.jsonl`). |
| `raw/` | no | GlotOCR-bench images + `ground_truth.jsonl` per language. |
| `predictions/` | no | Tesseract / Surya / PaddleOCR jsonl + `error_taxonomy.csv`. |
| `cache/` | no | Renderer crops, layouts, Sarvam Extract JSON (SHA-256 keyed). |
| `predictions_backup/` | no | Stale copy of predictions; do not cite. |

Do not invent a second layout. Probe scripts take `--data-root data` (or `OCR_DATA_ROOT`). Analyses read `data/probe_results/` and write `docs/`.

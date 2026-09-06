# Training and evaluation-set provenance

**Generated:** 2026-09-05  
**Rule:** cite a file; mark unrecoverable rather than guess.

---

## Attention-ablation prefix (Step 0a)

When encoder memory is zeroed, **Table 5’s agree/flip / KL decomposition is not teacher-forced on the ground-truth prefix, and it is not a free-run of the zero-memory decoder.** It re-scores the zero-memory decoder under the **full-memory greedy token path** (after BOS). Mean confidence under zero memory is a **separate free-run**.

Deciding lines, `src/probes/probe_attention_ablation.py`:

```10:18:src/probes/probe_attention_ablation.py
Method (inference only, existing hindi/natural checkpoints):
  1. generate() with full encoder memory.
  2. generate() again with encoder memory replaced by zeros *before*
     memory_projection (prior-only).
  3. For per-step KL / top-1 / prior-sufficiency, re-score the
     zero-memory decoder under the *full-memory token prefixes*
     (teacher forcing) so sequence divergence does not confound the
     distribution comparison. Independent zero-memory greedy still
     supplies mean_confidence_zero for the headline contrast.
```

```150:167:src/probes/probe_attention_ablation.py
    full = generate(
        model, tensor, tokenizer, device=device, return_full_probs=True,
    )
    zero_indep = generate(
        model, tensor, tokenizer, device=device, zero_encoder_memory=True,
    )
    # Re-score zero-memory under the full-memory token path (after BOS).
    forced = full["token_ids"][1:]
    zero_tf = generate(
        model,
        tensor,
        tokenizer,
        device=device,
        zero_encoder_memory=True,
        return_full_probs=True,
        force_next_ids=forced,
        max_len=len(forced),
    )
```

`generate.force_next_ids` appends those ids instead of argmax (`src/models/instrument/generate.py` lines 69–74, 115–133). Record field `"prefix_alignment": "teacher_force_zero_on_full_tokens"` (`probe_attention_ablation.py` line 214).

**Implication for Table 5:** agree/flip is a **per-context** comparison (same prefixes), but the context is the **full-memory generation**, not the reference string. It is not a teacher-forced GT diagnostic.

---

## Instrument training (Hindi / natural)

Sources unless noted: `src/models/instrument/train.py`, `encoder.py`, `decoder.py`; `DECISIONS.md` #65; `COLAB_RUNS.md`; `data/manifests/hindi_natural.jsonl`.

| Item | Value | Source |
|---|---|---|
| Corpus | Rendered Hindi **line crops**, glyph-frequency mode `natural` | `export_manifest_scaled.py`; manifests under `data/manifests/` |
| Corpus size | **2538** lines in `data/manifests/hindi_natural.jsonl` (`wc -l`, this checkout). Paper n-gram analysis uses **2491** training lines after excluding the 60 eval GT strings (`docs/paper_defensibility_stats.md` §7). Flattened 2707 / inverted 2872 lines (same `wc`). Design target was 100 pages/mode (`DECISIONS.md` #65; `export_manifest_scaled.py` `--pages-per-mode`). | manifests; stats §7; #65 |
| Text source for those pages | Ground-truth **strings** from GlotOCR Hindi (`data/raw/hindi/ground_truth.jsonl`), resampled into Tier A pages | `export_manifest_scaled.py` `load_corpus` |
| Font set | First hit among Kohinoor / Sangam / ITF (macOS) or Noto Sans Devanagari (Linux/Colab). **Which file Colab actually opened is unrecoverable** (no per-page `font_path` committed for the 100-page run). | `src/renderer/render.py` `FONT_CANDIDATES` |
| Augmentation pipeline | **None at train time.** `train.py` loads grayscale crops, `/255`, white pad. Training export is `render_tier_a` (mode only; no sampled degradation). Tier B blur/noise/skew exists in `degradation_profile.py` and is **not** the Hindi instrument training path. | `train.py` `collate_batch`; `render_tier_a` |
| Input resolution | Line crop **height 70 px** (5× patch 14). Width variable, padded per batch to a multiple of 14. Real eval images are **resized** to height 70 preserving aspect ratio (`LANCZOS`). | `export_line_manifest.py` `CANONICAL_LINE_HEIGHT`; `probe_utils.resize_to_canonical_height`; `DECISIONS.md` #44 |
| Patch size | 14 | `train.py` `PATCH_SIZE`; `encoder.py` |
| Encoder | 6 layers, `d_model=320`, 5 heads, mlp_ratio 4, dropout 0.1, sinusoidal positions, grayscale 1 channel | `encoder.py` |
| Decoder | 5 layers, `d_model=384`, 6 heads, mlp_ratio 4, dropout 0.1, learned positions, tied embedding/head | `decoder.py` |
| Bridge | `Linear(320 → 384)` | `train.py` `InstrumentModel` |
| Optimizer | AdamW | `train.py` (`torch.optim.AdamW`) |
| Learning rate | `3e-4` (CLI default) | `train.py` `--lr` |
| Schedule | **None.** Constant LR for all 5000 steps. | `train.py` `train()` — no scheduler |
| Batch size | 32 | `train.py` `--batch-size` |
| Total steps | 5000 | `train.py` `--total-steps`; Probe 3b snapshots at 500/1000/2000/3000/5000 |
| Mixed precision | fp16 autocast + GradScaler on CUDA; T4 has no bf16 | `train.py`; `IMPLEMENTATION.md` / `DECISIONS.md` #2 |
| Hardware | Free Colab **T4** | `COLAB_RUNS.md`; `AGENTS.md` |
| Wall-clock | **Unrecoverable** from this checkout. No training log with elapsed time is committed. Probe 3b records loss vs step, not GPU hours (`docs/probe3_curve_analysis.md`). | — |
| Checkpoint-selection criterion | **Last step of the 5000-step run.** `checkpoint_hindi_natural_seed{N}.pt` is overwritten every `--checkpoint-every` (200) steps; final file is step 5000. No val-loss selection. Probe 3b uses `--keep-snapshots` step files, not a different “best” ckpt. | `train.py` `checkpoint_path` / loop |

Parameter count at `|V|=367`: 19,607,104 (`DECISIONS.md` #65). Vocabulary is built from the run’s manifest with `min_freq=5` (`tokenizer.build_vocab`).

---

## 60 real Hindi scans (model-eval set)

| Item | Value | Source |
|---|---|---|
| Source | Hugging Face `cis-lmu/GlotOCR-bench`, config `Deva`, language code `hin_Deva`, split `test` | `src/data_pipeline/fetch_glotocr.py` |
| Count | 60 rows → 60 plain + 60 degraded PNGs | `probe5b_zeroshot_floor.py` docstring (lines 79–80); same jsonl |
| Scanner / capture device | **Unrecoverable here.** GlotOCR `source` field is copied into `ground_truth.jsonl` but that jsonl is gitignored (`data/raw/`). Dataset card / paper not re-fetched for this file. | `fetch_glotocr.py` writes `row["source"]` |
| Native resolution | **Unrecoverable** from git. Probe comment: real images “e.g. 254px” tall before resize (`probe_utils.py` `resize_to_canonical_height`). | comment only |
| Line crops | **Eval does not line-crop.** Probe 5b / ablation / GT-likelihood use `img_plain_path` (full GlotOCR page image), then `resize_to_canonical_height(..., 70)`. Training line crops are a **different** pipeline (HarfBuzz render → `export_line_crops`). | `probe5b` `resolve_image_path`; `export_line_manifest.py` |
| Sample for probes | `random.Random(0).sample(hindi_rows, min(n, 60))` = full pool of 60 when `n_samples≥60` | `probe_attention_ablation.build_hindi_sample`; `probe5b` |

---

## Ol Chiki and Perso-Arabic images

| Script | GlotOCR config | Language code | Count (probe docstring) | Source |
|---|---|---|---|---|
| Santhali / Ol Chiki | `Olck` | `sat_Olck` | 100 | `fetch_glotocr.py`; `probe5b_zeroshot_floor.py` |
| Kashmiri / Perso-Arabic | `Arab` | `kas_Arab` | 20 | same |

Same fetch: `img_plain` and `img_old_document` saved; probes use **plain**. Scanner/DPI: **unrecoverable** from git (raw jsonl not committed).

---

## Stage 0 engine table (180 / 222 / 10) vs the 60-image eval set

- **60-image set:** Hindi GlotOCR pages, used three times (seeds 0/1/2) for instrument probes. n=180 in Probe 5b Hindi is **60 images × 3 seeds**, not 180 pages (`docs/paper_defensibility_stats.md` Item 10 / panel note).
- **Stage 0 n** is **prediction rows**, not those 60×3 records. `paper_defensibility_stats.py` Follow-Up 7 loops `error_taxonomy.ENGINES` × `LANGUAGES` (`hindi`, `bengali`, `santhali`, `kashmiri`) × every non-skipped `data/predictions/{engine}/{language}.jsonl` row, each identified by `(id, variant)` with `variant` ∈ `{plain, degraded}` (`run_baselines.py`).
- Tesseract **n=180**, Surya **n=222**, PaddleOCR **n=10** are those totals after skip-null (`docs/paper_defensibility_stats.md` Follow-Up 7 table). Incomplete Paddle coverage is why n=10.
- Overlap with the 60 Hindi eval IDs: **not computed in this file.** Same GlotOCR Hindi pool is the intended source for Hindi Stage 0 rows, but Stage 0 also includes Bengali / Santhali / Kashmiri and both plain and degraded. Exact ID-set intersection needs `data/predictions/` + `data/raw/` (gitignored). Treat as **unrecovered in-repo** until those files are present.

---

## Grapheme segmenter for CER

Paper grapheme CER (`docs/paper_defensibility_stats.md` Item 1): Tier-1 normalize, then Unicode extended grapheme clusters.

- Library: third-party **`regex`**, pattern **`r"\X"`** (not stdlib `re`).
- Code: `src/analysis/paper_defensibility_stats.py` (`graphemes`); same primitive `src/models/instrument/tokenizer.py` `split_graphemes`; `src/analysis/make_paper_figures.py` `graphemes`.
- Version **on this laptop at generation of this doc:** `regex==2026.7.19` (`importlib.metadata`). **Colab’s pin at training/eval time is unrecoverable** (`requirements.txt` does not pin `regex`).
- Stage 0 engine table is **full-string** Tier 1 / Tier 2 / UNREVIEWED (`error_taxonomy.classify`), **not** grapheme CER. Do not conflate the two.

---

## Related but out of scope

Bengali instrument checkpoints: Colab zip only (`COLAB_RUNS.md`); not used in the paper’s mechanistic tables (`DECISIONS.md` #66).

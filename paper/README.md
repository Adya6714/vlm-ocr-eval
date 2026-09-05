# Paper sources

NeurIPS 2026 preprint sources live here — not under the template zip name
`NeurIPS 2026 Formatting Instructions/`. Spaces in that path break Make,
shell one-liners, and CI; this directory is the single compile root.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`\usepackage{neurips_2026}` loads `neurips_2026.sty` in this folder.
`\includegraphics{figures/...}` loads `paper/figures/`, which is also the
default `--paper-dir` of `src/analysis/make_paper_figures.py`. Do not copy
PDFs into a second tree.

Regenerate publication figures (repo root):

```bash
PYTHONPATH=src/eval python3 src/analysis/make_paper_figures.py \
  --results-root data/probe_results \
  --out-dir docs/figures \
  --paper-dir paper/figures \
  --seeds 0 1 2
```

Working PNGs (titles, grid) go to `docs/figures/` only. Figure 5 is working-only.

## Figure status

| File | In `main.tex` | Notes |
|---|---|---|
| `figures/fig1_position_dissociation.pdf` | yes | Top panel: uniform, trigram, 4-gram, 5-gram reference lines (LM values from `docs/paper_defensibility_stats.md` §7). Positions are 0-based tokens after `<BOS>` (`gt_force_ids` drops BOS). If a caption ever disagrees with “position 1 = second generated grapheme,” check that indexing before editing the table. |
| `figures/fig2_ablation_kl.pdf` | yes | |
| `figures/fig3_confidence_distributions.pdf` | yes | |
| `figures/fig4_regime_contrast.pdf` | yes | |
| Fig 5 heatmap | no | `docs/figures/fig5_output_degeneracy.png` only |

## Corrections already in `main.tex`

These are in the committed source (do not re-apply as if missing):

1. **n-gram claim reversal** — later positions are compared to 4-gram / 5-gram, not claimed sharper than a trigram.
2. **40+ bucket** — a second likelihood collapse past position 39 is reported and down-weighted.
3. **Blank-beats-real softening** — CER real vs blank is not treated as a robust blank advantage.
4. **Length mechanism** — CER inversion discussed via output length / matching.
5. **Flip-rate conflation** — per-image top-1 disagreement vs pooled per-step flip rate are distinguished.
6. **Tier 1 unreviewed fractions** — encoding-variant shares are lower bounds given UNREVIEWED mass.

## Still outstanding

- **Positive control** (open-weights Devanagari-competent model) still needs GPU; stays future work.
- **Mismatch TF, cross-attn norms, noise/scrambled** probes are authored; checkpoints are on Colab, not this laptop (`DECISIONS.md` #63).
- **Bibliography:** some entries are still TODO/UNVERIFIED in `refs.bib` if Overleaf overwrote the filled versions; restore from `DECISIONS.md` #67 before camera-ready.
- **Section 3 hyperparameters:** the short Instrument section may still omit layer/width/steps; the trained values are in `DECISIONS.md` #65–#66.

`checklist.tex` (NeurIPS main-track ethics fragment) is not in this folder. Re-add with `\input{checklist}` only if submitting to the NeurIPS main track.

## Overleaf

Point the project at `paper/` (this directory). A project still bound to
`NeurIPS 2026 Formatting Instructions/` must be re-linked after the rename.

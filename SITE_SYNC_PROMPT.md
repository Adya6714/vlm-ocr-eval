# Site sync (GitHub Pages `index.html`)

The live page has two modes — **Research** and **Production** — toggled in the top bar.
`paper/main.pdf` is canonical for preprint claims. Do not invent numbers.
Numbers live in `docs/paper_defensibility_stats.md`, `docs/position_matched_ngrams.md`,
Stage 5–6 markdown, and `paper/main.tex`. Cite those files; do not paraphrase
headline tables from memory.

## Critical framing

Not “VLMs ignore images.” Claim: in this instrument, confidence stays
high even when the model does not read held-out images.
Eval images are GlotOCR synthetic line renders, not phone photos.

Tab labels: **Research** / **Production**. Never title the Production tab
“Sarvam.” Sarvam Vision is named in Production body copy as the production
motivation and transfer reference — not as an accuracy competitor.

## Research tab (scientific paper story)

Audience: scientific research walkthrough — what / how / why / what’s next.
Reflect `BOOK.md` depth in plain language; keep flowcharts and figures.

Suggested scroll order:

1. Question / framing (hero)
2. Why (Indic OCR + three quantities)
3. Build pipeline (Stages 0→6 flowchart)
4. Measure (Tier 0/1/2)
5. Renderer (HarfBuzz, exposure dial)
6. Architecture (~19.6M instrument)
7. What 0.90 means (peak of ~367; entropy / rank)
8. Position 0 dissociation
9. **Full position profile** (0 / 1 / 2–39 / 40+)
10. Controls (blank / noise / scramble / ablation) + Probe 3 demo
11. Calibration trap (ECE / AUROC)
12. Related work
13. Findings / limits / next

## Production tab (Sarvam-motivated, not labeled Sarvam)

Audience: product / transfer conversation. Start with motivation from
published Indic OCR Bench gaps, then transfer results.

1. Motivation (Hindi 95.91 / Santhali 80.32 / Kashmiri 55.93)
2. Stage 5a gap (Δ confidence 0.0027)
3. Why instrument (API cannot run counterfactuals)
4. Mechanism recap (link Research)
5. Stage 5b Spearman null
6. Stage 6 triage (worse than random)
7. Pitch / next / talk track

## Animations (keep simple)

- Section reveal + hero rise
- Position-profile bars fill on scroll (`#posProfile.in`)
- Probe 3 meter width
- Respect `prefers-reduced-motion`

## Depth rule

Do **not** shorten to “elevator pitch only.” Both tabs should stay
BOOK-reflective: detailed, easy language, diagrams/flowcharts, clear
what-we-built / how / why / further work. Prefer expanding sections over
cutting them when syncing from BOOK.

## Numbers sources

- Position profile: `docs/position_matched_ngrams.md`
- Pos-0 / Probe 3 aggregates: `docs/paper_defensibility_stats.md`
- Entropy/rank: `paper/main.tex` Table E1
- 5a: `docs/sarvam_vision_confidence.md`
- 5b: `docs/stage5b_rank_correlation.md`
- 6: `docs/stage6_triage_cascade.md`

Push `index.html` to `main` before calls.

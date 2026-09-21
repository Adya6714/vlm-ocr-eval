# Site sync (GitHub Pages `index.html`)

The live page has two modes — **Production** (default, first) and **Research**
(second) — toggled in the top bar.
`paper/main.pdf` is canonical for preprint claims. Do not invent numbers.
Numbers live in `docs/paper_defensibility_stats.md`, `docs/position_matched_ngrams.md`,
Stage 5–6 markdown, and `paper/main.tex`. Cite those files; do not paraphrase
headline tables from memory.

## Critical framing

Not “VLMs ignore images.” Claim: in this instrument, confidence stays
high even when the model does not read held-out images.
Eval images are GlotOCR synthetic line renders, not phone photos.

Tab labels: **Production** (first, default) / **Research** (second). Never title the Production tab
“Sarvam.” Sarvam Vision is named in Production body copy as the production
motivation and transfer reference — not as an accuracy competitor.

## Research tab (scientific paper story)

Audience: scientific research walkthrough — what / how / why / what’s next.
Reflect `BOOK.md` depth in plain language; keep flowcharts and figures.

**Talk-first scroll order** (CSS `--paper-order`):

1. Question / framing (hero) — “scroll like a talk”
2. Why (Indic OCR + three quantities)
3. What 0.90 means (animated softmax viz)
4. Position 0 dissociation (`num-pop` + `dissoc-grid`)
5. Full position profile (animated bars)
6. Controls — **two interactives**: Probe 3 confidence (text/blank/noise) + Tier 0d log p(GT) (adds scramble)
7. 60-second “say this”
8. Then build / measure / renderer / architecture / calibration / findings / limits / next

Mermaid is **not** used on the live site (CDN never loaded reliably on
Pages). Flowcharts are native `.flow-board` / `.flow-pipe` / `.flow-node`
HTML+CSS cards — always render, match the site chrome. Do not reintroduce
`<pre class="mermaid">` blocks.

Keep animations: hero rise, gap fills, pos-profile, soft-viz, freq-grid, glyph invite, meter fill, scroll reveal. Respect `prefers-reduced-motion`.

## Production tab (default · Sarvam-motivated, not labeled Sarvam)

Audience: product / transfer conversation. Start with motivation from
published Indic OCR Bench gaps, then transfer results. Include a mini
Probe 3 interactive on the mechanism section so the call can demo without
flipping tabs mid-sentence.

1. Motivation (Hindi 95.91 / Santhali 80.32 / Kashmiri 55.93)
2. Stage 5a gap (Δ confidence 0.0027)
3. Why instrument (API cannot run counterfactuals)
4. Mechanism recap + Probe 3 demo
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

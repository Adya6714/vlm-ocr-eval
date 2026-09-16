# Site sync (GitHub Pages `index.html`)

The live page is a **call walkthrough** (Diagnosis / Sarvam tabs).
`paper/main.pdf` is canonical for preprint claims. Do not invent numbers.

## Critical framing

Not “VLMs ignore images.” Claim: in this instrument, confidence stays
high even when the model does not read held-out images.

## Diagnosis tab flow (scroll as a talk)

1. Claim / framing
2. What 0.90 means (peak of 367; Token A varies; sharp vs uncertain;
   entropy 0.333, median rank 70.5, 35% below rank 100 — from paper)
3. Position 0 dissociation
4. **Full position profile** (0 / 1 / 2–39 / 40+) — not pos-0 only
5. Blank & ablation (careful phrasing) + interactive Probe 3
6. Say-this script
7. Next priorities

## Sarvam tab

Production motivation → 5a gap → why instrument → 5b/6 → pitch.

## Animations (keep simple)

- Existing section reveal + hero rise
- Position-profile bars fill on scroll (`#posProfile.in`)
- Probe 3 meter width
- Respect `prefers-reduced-motion`

## Numbers

Position profile: `docs/position_matched_ngrams.md`.
Pos-0: `docs/paper_defensibility_stats.md`.
Entropy/rank: `paper/main.tex` Table E1.

Push `index.html` to `main` before calls.

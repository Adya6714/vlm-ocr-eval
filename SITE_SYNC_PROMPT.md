# Site sync (GitHub Pages `index.html`)

Run this whenever `paper/main.tex` changes claims, terminology, or a
headline number. The Preprint tab is a reading aid; `paper/main.pdf` is
canonical. Do not invent numbers. Cite `docs/paper_defensibility_stats.md`
or the tex.

## Terminology (our data)

- **text-bearing** — an evaluation image containing rendered text (vs **blank**).
- **held-out** — the evaluation set generally.
- **real** — only other people’s data (Singh’s scans, GlotOCR source
  sentences, published scan corpora). Never our Probe 5b/6 images.
  Those are GlotOCR *renders*, not photographs.

## Position 0

Do **not** anchor geometric-mean \(p(\mathrm{GT})\) on uniform
\(1/|V|\approx 2.725\times10^{-3}\). The paper retracts that comparison
(softmax sharpness, not specifically disfavouring the truth).

Use the position-matched \(n\)-gram: instrument \(\approx -24.54\) nats
vs text-only first-symbol marginal \(\approx -4.73\) (19.8 nats / more
than eight orders below a **text-only** baseline). Pair with
self-generated max-softmax \(\approx 0.90\).

## Training contamination

The instrument trains on the **full** 2,538-line `hindi_natural.jsonl`.
`train.py` has no eval-string filter. 19 of 60 evaluation strings appear
verbatim as training lines (47 manifest rows). State that on the
Preprint tab. Position 0/1 still fail at floor (contamination can only
have helped). Do not treat mid-sequence \(-0.15\) or synthetic AUROC
0.838 as a clean held-out result.

## Parameter count

\(\approx 19.6\mathrm{M}\) at \(|V|=367\) (19,607,104), not 19.5M.

## After edits

Push `index.html` (and this file) to `main` so Pages updates before any
external send that links the site.

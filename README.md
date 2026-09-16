# vlm-ocr-eval

This was **not primarily an OCR-model-building project**. It is a controlled investigation into whether OCR decoder confidence reflects **visual evidence** — or only whether the generated text looks linguistically plausible.

**Critical framing:** The paper does **not** prove that a model that can read is hallucinating. It shows that, in this particular ~19.6M from-scratch **instrument**, confidence stays extremely high even when the model demonstrably does **not** read the evaluation image. Never claim “VLMs don’t look at images.”

Production pipelines often route on confidence. This repo asks what happens when confidence stays near ceiling while teacher-forced *p*(ground truth) at position 0 is ~10⁻¹¹.

| | |
|---|---|
| Preprint | [*Reading Without Looking*](paper/main.pdf) · [source](paper/main.tex) |
| **Call walkthrough** | [adya6714.github.io/vlm-ocr-eval](https://adya6714.github.io/vlm-ocr-eval/) — **Research** / **Production** tabs |
| Full reference | [`BOOK.md`](BOOK.md) — first principles, related work, Sarvam pitch, interview Q&A |
| Measurements | [`docs/RESULTS.md`](docs/RESULTS.md) · [`docs/paper_defensibility_stats.md`](docs/paper_defensibility_stats.md) |
| Training setup | [`docs/training_config.md`](docs/training_config.md) |
| Sarvam Extract (not in PDF) | [`docs/sarvam_vision_confidence.md`](docs/sarvam_vision_confidence.md) |

The probe model is a **measurement instrument** (19,607,104 params at |V|≈367), trained from scratch with no Indic pretraining — not a production OCR engine and not a Sarvam competitor.

---

## One picture

```text
PROBLEM — How trustworthy is OCR confidence?
   ↓
Stage 0 — Indic-aware evaluation (what counts as “wrong”?)
   ↓
Stage 1 — Controlled renderer (exposure dial)
   ↓
Stage 2 — From-scratch instrument (empty Indic history)
   ↓
Probes — blank/noise · unseen scripts · ablation · position-0 p(GT) · n-gram
   ↓
Sarvam — Does confidence track published difficulty? Does ranking transfer?
   ↓
Triage — Can confidence route pages? (worse than random at 20%)
```

**Headline (instrument):** position 0 — max-softmax ≈ 0.90 while *p*(correct first grapheme) ≈ 2.2×10⁻¹¹; GT never argmax (0/180); blank similar. **Full profile:** pos 1 still poor; positions 2–39 recover toward a text-only 4–5-gram (~91–94% argmax agreement). Cite [`docs/paper_defensibility_stats.md`](docs/paper_defensibility_stats.md) · [`docs/position_matched_ngrams.md`](docs/position_matched_ngrams.md).

**Sarvam framing:** production reference for “does confidence track difficulty?” — not an accuracy bake-off. Published bench: Hindi 95.91% / Santhali 80.32% / Kashmiri 55.93% ([sarvam.ai/blogs/sarvam-vision](https://www.sarvam.ai/blogs/sarvam-vision)).

---

## Correctness ≠ confidence ≠ grounding

```text
CORRECTNESS — right text?                         → often LOW here
GROUNDING   — did the image support the output? → LOW
CONFIDENCE  — how peaked is the softmax?        → HIGH
```

Max-softmax only measures the third.

---

## Findings (orientation only — cite docs)

| Claim | Result |
|---|---|
| Position 0 | *p*(GT) ~10⁻¹¹ vs max-softmax ~0.90; 0/180 argmax |
| Ablation | Δ conf ≈ −0.003; ~8% flips carry ~97% of KL |
| Mid-sequence | ≈ 4–5-gram text-only LM |
| Probe 1 | Dial worked; flat/inverted at floor → β withheld |
| Calibration | ECE ≈ 0.811; AUROC 0.838 **in-manifest only**; held-out graded ≈ 0.570 |
| Sarvam 5a | ~40 pp accuracy gap; confidence Δ 0.0027 |
| Transfer 5b | ρ = 0.0293, p = 0.8267 (null) |
| Triage 6 | Instrument confidence worse than random at 20% |

---

## What not to claim

| Don’t say | Say instead |
|---|---|
| “We proved VLMs don’t look at images.” | Confidence without grounding in **this** instrument. |
| “We trained production OCR.” | ~19.6M research instrument. |
| “18.3% held-out accuracy.” | Training-manifest synthetic regime. |
| “GlotOCR = real scans.” | Eval images are **renders**. |
| “Sarvam doesn’t look at images.” | Page-level confidence didn’t track published language spread; ranking didn’t predict Sarvam CER. |

Biggest gap: **positive control** (known-good AR Devanagari reader under the same position-0 protocol).

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make smoke-test
pytest -q
```

Further: [`BOOK.md`](BOOK.md) · [`IMPLEMENTATION.md`](IMPLEMENTATION.md)

## Citation

```bibtex
@misc{srivastava2026reading,
  title  = {Reading Without Looking: A Position-Resolved Profile of
            Confidence and Correctness in a Small Document OCR Model},
  author = {Srivastava, Adya},
  year   = {2026},
  url    = {https://github.com/Adya6714/vlm-ocr-eval}
}
```

Raw GlotOCR pages are not redistributed. This repository is not an OCR leaderboard.

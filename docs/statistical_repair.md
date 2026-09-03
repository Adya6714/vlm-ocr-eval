# Statistical repair — Probe 5b and Probe 3b

**Date:** 2026-09-03 (updated after seeds 1–2 landed)  
**Code:** `src/analysis/analyze_probe5b.py`, `src/analysis/analyze_probe3_curve.py`  
**Decisions:** #14 (three seeds), #51 (Bonferroni column), #52 (TOST δ = 0.05),
#53 (Kashmiri retraction)  
**Inputs:** `data/probe_results/probe5b_hindi_natural_seed{0,1,2}.jsonl`,
`data/probe_results/probe3_curve_hindi_natural_seed{0,1,2}.json`

Claim-facing summaries:
[`probe5b_analysis.md`](probe5b_analysis.md),
[`probe3_curve_analysis.md`](probe3_curve_analysis.md).

---

## 1. Non-independence → cluster bootstrap of images

Cluster bootstrap (n_boot = 10_000, resample **images**) sits beside the
naive Bonferroni z so SE inflation is visible. On these runs, once the
unit is already the image, naive SE ≈ bootstrap SE (~1×). The procedure
still gates claims; tokens were never a valid *n*.

---

## 2. Equivalence — lead with range vs seed-SD, TOST second

### Threshold-free lead claim (3 seeds)

Across-seed means of image-level mean confidence:

| Condition | seed0 | seed1 | seed2 | Mean | SD |
|-----------|-------|-------|-------|------|----|
| hindi | 0.9824 | 0.9897 | 0.9899 | 0.9873 | **0.0043** |
| santhali | 0.9848 | 0.9847 | 0.9876 | 0.9857 | 0.0016 |
| kashmiri | 0.9901 | 0.9894 | 0.9887 | 0.9894 | 0.0007 |
| blank | 0.9814 | 0.9933 | 0.9824 | 0.9857 | **0.0066** |

**Between-condition range** of those across-seed means:
0.9857 → 0.9894 = **0.0037**.

That range is **smaller** than hindi's across-seed SD (0.0043) and
blank's (0.0066). Condition means sit closer to each other than a
single condition jitters across training seeds. This needs **no
assumed effect-size threshold** — it is the primary equivalence
argument. TOST at δ = 0.05 (DECISIONS.md #52) still passes on every
seed × contrast and remains the formal test; range-vs-SD is the more
compelling plain-language case.

### TOST (retained, secondary)

Every seed × {santhali, kashmiri, blank} contrast has bootstrap 90% CI
of Δ inside [−0.05, +0.05]. Details in `probe5b_analysis.md` §2.

---

## 3. Kashmiri Bonferroni claim — RETRACTED

| Seed | Hindi mean | Kashmiri mean | Δ (kas−hin) | Naive z | \|z\| ≥ 2.39? |
|------|------------|---------------|-------------|---------|---------------|
| 0 | 0.9824 | 0.9901 | +0.0077 | **2.538** | **yes** |
| 1 | 0.9897 | 0.9894 | −0.0003 | −0.110 | no |
| 2 | 0.9899 | 0.9887 | −0.0012 | −0.487 | no |

Seed-0 alone looked like a corrected significant elevation of Kashmiri
confidence over Hindi. Seed 1 **reverses the sign** (hindi 0.9897 >
kashmiri 0.9894). Only 1/3 seeds clear Bonferroni.

**Retraction (DECISIONS.md #53):** do not cite Kashmiri confidence as
significantly above Hindi. The seed-0 z = 2.54 was a **single-seed
artifact caught by the three-seed requirement** (DECISIONS.md #14).
What it cost to catch: two additional full Probe 5b inference passes
(seeds 1–2) on the same hindi/natural checkpoints — cheap relative to
shipping a non-replicating claim.

---

## 4. Distributions / ceilings

Per-seed histograms and P(>0.95) / P(>0.99) remain in
`probe5b_analysis.md` §2. Means near 0.98–0.99 still sit on ceiling
mass; micro-deltas are not an abstention signal.

---

## 5. Script substitution — 360/360

| Seed | Santhali zero/n | Kashmiri zero/n | Total |
|------|-----------------|-----------------|-------|
| 0 | 100/100 | 20/20 | 120/120 |
| 1 | 100/100 | 20/20 | 120/120 |
| 2 | 100/100 | 20/20 | 120/120 |
| **all** | **300/300** | **60/60** | **360/360** |

Every unseen-script image across all three seeds emitted **zero**
graphemes of the script in the image (fluent Devanagari instead). This
is the sharp Probe 5b finding; confidence contrasts are not.

---

## 6. Probe 3b — gap indistinguishable from zero across seeds

| Step | Gap seed0 | seed1 | seed2 | Mean | SD | Sign flip? | \|SD\|>\|mean\|? |
|------|-----------|-------|-------|------|----|------------|------------------|
| 500 | +0.0097 | +0.0248 | −0.0129 | +0.0072 | 0.0190 | yes | yes |
| 1000 | +0.0019 | +0.0072 | −0.0103 | −0.0004 | 0.0090 | yes | yes |
| 2000 | +0.0017 | −0.0027 | +0.0056 | +0.0015 | 0.0041 | yes | yes |
| 3000 | −0.0082 | −0.0102 | −0.0040 | −0.0075 | 0.0031 | no | no |
| 5000 | +0.0162 | −0.0038 | −0.0073 | +0.0017 | 0.0126 | yes | yes |

Sign flips at **4 of 5** steps; |SD| exceeds |mean| at **4 of 5**
steps. It is now defensible to call the real−blank gap
**indistinguishable from zero** across training seeds.

**Step 3000 observation (not over-claimed):** negative in all three
seeds (mean −0.0075, magnitude 0.0075). Stable single-step sign; does
**not** license calling the whole curve a blank>real reversal.

The earlier single-seed "consistent with noise" wording is superseded:
with three seeds we can say the stronger thing (≈ 0) without pretending
seed-0's step-3000 dip was already known to be noise.

Headline unchanged: loss collapses while confidence rises and accuracy
stays near floor — undertraining and ungrounded confidence are both
true; confidence tracks training progress, not image evidence.

---

## 7. How to regenerate

```bash
PYTHONPATH=src python3 src/analysis/analyze_probe5b.py
PYTHONPATH=src python3 src/analysis/analyze_probe3_curve.py
```

Seeds are discovered under `data/probe_results/` as
`probe5b_*_seed{0,1,2}.jsonl` and `probe3_curve_*_seed{0,1,2}.json`.

# Probe 3b analysis — training curve (real vs blank confidence)

**Generated:** 2026-09-03  
**Sources:** `data/probe_results/probe3_curve_hindi_natural_seed0.json`, `data/probe_results/probe3_curve_hindi_natural_seed1.json`, `data/probe_results/probe3_curve_hindi_natural_seed2.json`  
**Run:** hindi/natural/seeds [0, 1, 2]  
**Snapshots:** 5 steps [500, 1000, 2000, 3000, 5000]  
**Samples per step:** 30  
**Correction note:** see [statistical_repair.md](statistical_repair.md).

---

## 1. Across-seed curve table

| Step | Loss (mean) | Real conf (mean) | Blank conf (mean) | Gap mean | Gap SD | Sign flip? | |SD|>|mean|? | Acc (mean) |
|------|-------------|------------------|-------------------|----------|--------|------------|--------------|------------|
| 500 | 3.2102 | 0.8906 | 0.8833 | +0.0072 | 0.0190 | yes | yes | 0.0000 |
| 1000 | 1.0346 | 0.9693 | 0.9697 | -0.0004 | 0.0090 | yes | yes | 0.0000 |
| 2000 | 0.6019 | 0.9784 | 0.9769 | +0.0015 | 0.0041 | yes | yes | 0.0000 |
| 3000 | 0.3553 | 0.9882 | 0.9956 | -0.0075 | 0.0031 | no | no | 0.0222 |
| 5000 | 0.1822 | 0.9925 | 0.9907 | +0.0017 | 0.0126 | yes | yes | 0.1333 |

### 1b. Per-seed gaps

| Step | seed0 | seed1 | seed2 |
|------|------|------|------|
| 500 | +0.0097 | +0.0248 | -0.0129 |
| 1000 | +0.0019 | +0.0072 | -0.0103 |
| 2000 | +0.0017 | -0.0027 | +0.0056 |
| 3000 | -0.0082 | -0.0102 | -0.0040 |
| 5000 | +0.0162 | -0.0038 | -0.0073 |

## 2. Interpretation

Across seeds [0, 1, 2], loss fell ~18× (3.210 → 0.182) while mean real confidence **rose** (0.891 → 0.992) and accuracy stayed near floor until late (mean acc 0.133 at step 5000).

The real−blank gap **sign flips across seeds at 4 of 5 steps**, and |gap SD| exceeds |gap mean| at 4 of 5 steps. It is now defensible to call the gap **indistinguishable from zero** across training seeds — not an emerging vision signal, and not a reliable negative either.

**Step 3000 observation (not over-claimed):** gap is negative in **all 3 seeds** (-0.0082, -0.0102, -0.0040; mean -0.0075, magnitude 0.0075). Recorded as a stable single-step sign; it does not license calling the whole curve a blank>real reversal.

### Correct framing

Undertraining and ungrounded confidence are **not** competing explanations. The model **is** undertrained (accuracy low on this curve). The finding is that its confidence gives **no indication** of that: a well-calibrated undertrained model would report **low** confidence. This one reports ~0.99 and rises as training proceeds.

The in-script interpretation that hedges between "(a) ungrounded" and "(b) undertrained" therefore misses the point. Both are true at once; the calibration failure is that confidence tracks training progress (loss ↓, conf ↑) instead of image evidence (gap ≈ 0 across seeds).

### What this does not establish

- That longer training would never open a real−blank gap (only that through 5000 steps × 3 seeds it does not).
- That the same curve holds for flattened/inverted conditions.
- That production OCR confidence is similarly ungrounded (instrument only).

# Probe 5b analysis — zero-shot confidence floor

**Generated:** 2026-09-03  
**Sources:** `data/probe_results/probe5b_hindi_natural_seed0.jsonl`, `data/probe_results/probe5b_hindi_natural_seed1.jsonl`, `data/probe_results/probe5b_hindi_natural_seed2.jsonl`  
**Run:** hindi/natural/seeds [0, 1, 2]  
**Records:** 720 across 3 seed(s)  
**Method:** per-seed image-level means; naive pairwise z (Bonferroni α = 0.05/3 = 0.016667, critical |z| = 2.394) beside cluster-bootstrap CIs (n_boot = 10000); TOST at δ = 0.05 (DECISIONS.md #52). Across-seed mean±SD of per-seed means — never pooled images. Full correction: [statistical_repair.md](statistical_repair.md).

Accuracy is **not** scored on Santhali or Kashmiri — the instrument's vocabulary is Devanagari grapheme clusters, so CER against Ol Chiki / Perso-Arabic ground truth would measure tokenizer impossibility, not vision failure (DECISIONS.md #50).

---

## 1. Across-seed confidence (primary table)

Per-seed means, then across-seed mean ± SD. This is the unit Decision #14 requires — not a pooled mega-sample.

| Condition | seed0 | seed1 | seed2 | Mean | SD |
|-----------|------|------|------|------|----|
| hindi | 0.9824 | 0.9897 | 0.9899 | 0.9873 | 0.0043 |
| santhali | 0.9848 | 0.9847 | 0.9876 | 0.9857 | 0.0016 |
| kashmiri | 0.9901 | 0.9894 | 0.9887 | 0.9894 | 0.0007 |
| blank | 0.9814 | 0.9933 | 0.9824 | 0.9857 | 0.0066 |

### 1a. Equivalence without an assumed δ (lead claim)

Between-condition range of across-seed means: **0.9857** (santhali) to **0.9894** (kashmiri) = **0.0037**.

Within-condition across-seed SD: hindi **0.0043**, blank **0.0066**, santhali **0.0016**, kashmiri **0.0007**.

The between-condition range is **smaller** than the within-condition seed noise for hindi and for blank. Condition means sit closer to each other than a single condition jitters across training seeds — that is the threshold-free case for treating zero-shot / blank confidence as equivalent to in-distribution confidence. TOST at δ = 0.05 (below) agrees, but this comparison does not need an assumed effect-size threshold.

### 1b. Kashmiri Bonferroni claim — RETRACTED

| Seed | Hindi mean | Kashmiri mean | Δ (kas−hin) | Naive z | |z| ≥ crit? |
|------|------------|---------------|-------------|---------|------------|
| 0 | 0.9824 | 0.9901 | 0.0077 | 2.538 | yes |
| 1 | 0.9897 | 0.9894 | -0.0003 | -0.110 | no |
| 2 | 0.9899 | 0.9887 | -0.0012 | -0.487 | no |

**Retraction (DECISIONS.md #53):** seed-0 naive z = 2.538 cleared Bonferroni, but the sign does **not** replicate. On seed 1, hindi mean (0.9897) **exceeds** kashmiri (0.9894). Only 1/3 seeds clear the corrected threshold. The seed-0 result was a **single-seed artifact caught by the three-seed requirement** (DECISIONS.md #14). Do not cite Kashmiri confidence as significantly above Hindi.

### 1c. Script substitution

**360/360** Santhali+Kashmiri images across all seeds emitted **zero** graphemes of the script visible in the image (fluent Devanagari instead). Substitution replicated on every unseen-script image.

| Seed | Santhali zero/n | Kashmiri zero/n | Total |
|------|-----------------|-----------------|-------|
| 0 | 100/100 | 20/20 | 120/120 |
| 1 | 100/100 | 20/20 | 120/120 |
| 2 | 100/100 | 20/20 | 120/120 |

### 1d. Per-seed Δ vs Hindi (mean ± SD across seeds)

| Contrast | seed0 | seed1 | seed2 | Mean Δ | SD |
|----------|------|------|------|--------|----|
| santhali − hindi | 0.0024 | -0.0051 | -0.0023 | -0.0016 | 0.0038 |
| kashmiri − hindi | 0.0077 | -0.0003 | -0.0012 | 0.0021 | 0.0049 |
| blank − hindi | -0.0010 | 0.0036 | -0.0075 | -0.0016 | 0.0056 |

## 2. Per-seed detail (naive z, bootstrap, TOST, ceilings)

### Seed 0

Source: `data/probe_results/probe5b_hindi_natural_seed0.jsonl` — 240 records.

| Condition | n | Mean | Naive 95% CI | Boot 95% CI | P(>0.95) | P(>0.99) |
|-----------|---|------|--------------|-------------|---------|----------|
| hindi | 60 | 0.9824 | [0.9774, 0.9874] | [0.9772, 0.9870] | 0.933 | 0.433 |
| santhali | 100 | 0.9848 | [0.9821, 0.9875] | [0.9820, 0.9874] | 0.980 | 0.460 |
| kashmiri | 20 | 0.9901 | [0.9869, 0.9932] | [0.9868, 0.9930] | 1.000 | 0.550 |
| blank | 60 | 0.9814 | [0.9782, 0.9846] | [0.9783, 0.9845] | 1.000 | 0.283 |

| Contrast | Δ | Naive z | |z|≥crit? | Boot 95% CI | Boot 90% CI | TOST equiv? |
|----------|---|---------|----------|-------------|------------|-------------|
| santhali − hindi | 0.0024 | 0.842 | no | [-0.0031, 0.0082] | [-0.0022, 0.0072] | yes |
| kashmiri − hindi | 0.0077 | 2.538 | yes | [0.0020, 0.0138] | [0.0028, 0.0128] | yes |
| blank − hindi | -0.0010 | -0.322 | no | [-0.0066, 0.0053] | [-0.0058, 0.0041] | yes |

Histogram counts (image-level mean_confidence):

- **hindi:** [0.000,0.900]:1, [0.900,0.950]:3, [0.950,0.970]:7, [0.970,0.980]:5, [0.980,0.990]:18, [0.990,0.995]:9, [0.995,1.000]:17
- **santhali:** [0.000,0.900]:0, [0.900,0.950]:2, [0.950,0.970]:10, [0.970,0.980]:12, [0.980,0.990]:30, [0.990,0.995]:23, [0.995,1.000]:23
- **kashmiri:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:0, [0.970,0.980]:2, [0.980,0.990]:7, [0.990,0.995]:6, [0.995,1.000]:5
- **blank:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:6, [0.970,0.980]:20, [0.980,0.990]:17, [0.990,0.995]:8, [0.995,1.000]:9

### Seed 1

Source: `data/probe_results/probe5b_hindi_natural_seed1.jsonl` — 240 records.

| Condition | n | Mean | Naive 95% CI | Boot 95% CI | P(>0.95) | P(>0.99) |
|-----------|---|------|--------------|-------------|---------|----------|
| hindi | 60 | 0.9897 | [0.9876, 0.9919] | [0.9876, 0.9918] | 1.000 | 0.567 |
| santhali | 100 | 0.9847 | [0.9824, 0.9869] | [0.9823, 0.9869] | 0.980 | 0.360 |
| kashmiri | 20 | 0.9894 | [0.9847, 0.9942] | [0.9844, 0.9937] | 1.000 | 0.550 |
| blank | 60 | 0.9933 | [0.9912, 0.9955] | [0.9911, 0.9953] | 1.000 | 0.800 |

| Contrast | Δ | Naive z | |z|≥crit? | Boot 95% CI | Boot 90% CI | TOST equiv? |
|----------|---|---------|----------|-------------|------------|-------------|
| santhali − hindi | -0.0051 | -3.168 | yes | [-0.0082, -0.0020] | [-0.0077, -0.0025] | yes |
| kashmiri − hindi | -0.0003 | -0.110 | no | [-0.0058, 0.0045] | [-0.0048, 0.0038] | yes |
| blank − hindi | 0.0036 | 2.335 | no | [0.0006, 0.0066] | [0.0011, 0.0061] | yes |

Histogram counts (image-level mean_confidence):

- **hindi:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:1, [0.970,0.980]:7, [0.980,0.990]:18, [0.990,0.995]:15, [0.995,1.000]:19
- **santhali:** [0.000,0.900]:0, [0.900,0.950]:2, [0.950,0.970]:11, [0.970,0.980]:11, [0.980,0.990]:40, [0.990,0.995]:20, [0.995,1.000]:16
- **kashmiri:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:1, [0.970,0.980]:1, [0.980,0.990]:7, [0.990,0.995]:2, [0.995,1.000]:9
- **blank:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:0, [0.970,0.980]:7, [0.980,0.990]:5, [0.990,0.995]:7, [0.995,1.000]:41

### Seed 2

Source: `data/probe_results/probe5b_hindi_natural_seed2.jsonl` — 240 records.

| Condition | n | Mean | Naive 95% CI | Boot 95% CI | P(>0.95) | P(>0.99) |
|-----------|---|------|--------------|-------------|---------|----------|
| hindi | 60 | 0.9899 | [0.9869, 0.9929] | [0.9866, 0.9928] | 1.000 | 0.633 |
| santhali | 100 | 0.9876 | [0.9852, 0.9899] | [0.9852, 0.9899] | 1.000 | 0.530 |
| kashmiri | 20 | 0.9887 | [0.9851, 0.9923] | [0.9850, 0.9922] | 1.000 | 0.500 |
| blank | 60 | 0.9824 | [0.9784, 0.9863] | [0.9783, 0.9860] | 1.000 | 0.483 |

| Contrast | Δ | Naive z | |z|≥crit? | Boot 95% CI | Boot 90% CI | TOST equiv? |
|----------|---|---------|----------|-------------|------------|-------------|
| santhali − hindi | -0.0023 | -1.181 | no | [-0.0060, 0.0016] | [-0.0054, 0.0010] | yes |
| kashmiri − hindi | -0.0012 | -0.487 | no | [-0.0059, 0.0036] | [-0.0052, 0.0028] | yes |
| blank − hindi | -0.0075 | -2.987 | yes | [-0.0124, -0.0026] | [-0.0118, -0.0034] | yes |

Histogram counts (image-level mean_confidence):

- **hindi:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:5, [0.970,0.980]:7, [0.980,0.990]:10, [0.990,0.995]:10, [0.995,1.000]:28
- **santhali:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:10, [0.970,0.980]:11, [0.980,0.990]:26, [0.990,0.995]:17, [0.995,1.000]:36
- **kashmiri:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:1, [0.970,0.980]:2, [0.980,0.990]:7, [0.990,0.995]:5, [0.995,1.000]:5
- **blank:** [0.000,0.900]:0, [0.900,0.950]:0, [0.950,0.970]:12, [0.970,0.980]:3, [0.980,0.990]:16, [0.990,0.995]:28, [0.995,1.000]:1

## 3. Finding (plain language)

Across seeds [0, 1, 2], mean confidence stays high on Hindi, Santhali, Kashmiri, and blank alike. The **lead** equivalence claim is threshold-free: between-condition range of across-seed means is 0.0037, smaller than hindi's across-seed SD (0.0043) and blank's (0.0066). TOST at δ = 0.05 passes on every seed × contrast, but the range-vs-SD comparison is the more compelling evidence because it needs no assumed effect-size threshold.

The seed-0 Kashmiri Bonferroni pass (z ≈ 2.54) is **retracted** — it does not replicate (DECISIONS.md #53). Three seeds earned their keep (DECISIONS.md #14).

Charset composition remains the sharp signal: **360/360** unseen-script images emitted zero correct-script characters — the model writes fluent Devanagari instead.

**What this does not establish:** that production OCR APIs behave identically (instrument only); that more Hindi training steps would fix zero-shot calibration (Probe 3b speaks to undertraining on in-distribution data, not unseen scripts); or that confidence differences of a few thousandths are useful for routing.

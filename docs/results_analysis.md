# Hindi probe results — factual inspection report

**Inspection date:** 2026-09-02  
**Source:** `Hindi Probe Results Final.zip` (extracted to `data/probe_results/`)  
**Method:** Direct file inventory + recomputation from raw jsonl (Probe 3 line-level means, Probe 5 bucket/ECE from `records`). No probe result files were modified.

**Not in this zip:** `probe3_curve_*.json` (training-curve probe), `probe5b_*.jsonl` (zero-shot floor).  
**Separate archive:** `Bengali Experiment Final.zip` holds Bengali checkpoints/manifests/baselines only — no probe jsonl files (see §6).

---

## Executive summary

All **9/9** expected Hindi run pairs (probe3 + probe5 × 3 conditions × 3 seeds) are present. Each run uses **n=100** manifest line-crops.

**What the numbers show:**

1. **Natural condition:** Mean confidence ≈ **0.99** on real, blank, and noise alike; real−blank gap ≈ **+0.004** (statistically above zero by seed SD, but **negligible in magnitude**). Accuracy ≈ **17–24%** per seed while confidence stays in the 0.90–1.00 bucket. Severe miscalibration (ECE ≈ **0.81** pooled).

2. **Flattened / inverted conditions:** The exposure dial **did move measurable quantities**. Mean confidence drops to ≈ **0.60** (flattened) and ≈ **0.46** (inverted). Real−blank gap becomes **negative** (blank/noise *higher* than real). Accuracy ≈ **0–1%** with confidence still often 0.4–0.7.

3. **Cross-condition:** Condition changes confidence, calibration, and accuracy dramatically. Natural is the pathological “~99% confident, ~18% accurate” regime; flattened/inverted reduce overconfidence but do not fix grounding (negative real−blank gap persists).

**What would be needed for stronger claims:** Probe 3 training curve (not run) to separate undertraining from structural ungrounded confidence on natural; Probe 5b (not run) for zero-script exposure; formal tests with n=3 seeds are weak — effects are large for condition comparisons but seed variance matters for gap sign tests.

---

## 1. Inventory

### 1.1 Files present (`Hindi Probe Results Final.zip`)

| File pattern | Count | Lines each (probe3 / probe5) |
|--------------|-------|------------------------------|
| `probe3_hindi_{condition}_seed{0,1,2}.jsonl` | 9 | 100 lines (one crop per line) |
| `probe5_hindi_{condition}_seed{0,1,2}.jsonl` | 9 | 1 JSON object; 100 `records` each |
| `summary_hindi.json` | 1 | Pooled 3-seed aggregates |

### 1.2 Complete run pairs (Probe 3 + Probe 5)

| Condition | Seed 0 | Seed 1 | Seed 2 |
|-----------|--------|--------|--------|
| **natural** | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| **flattened** | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| **inverted** | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |

**Complete pairs: 9 / 9.**

### 1.3 Missing from zip

| Item | Status |
|------|--------|
| `probe3_curve_*.json` | Not present |
| `probe5b_*.jsonl` | Not present |

---

## 2. Probe 3 (blank/noise control)

Per-line mean confidence averaged within each run; gap = mean(real) − mean(blank).

### 2.1 Per condition × seed

| Condition | Seed | n | Real | Blank | Noise | Real−Blank |
|-----------|------|---|------|-------|-------|------------|
| natural | 0 | 100 | 0.9935 | 0.9926 | 0.9896 | +0.0010 |
| natural | 1 | 100 | 0.9931 | 0.9870 | 0.9890 | +0.0061 |
| natural | 2 | 100 | 0.9954 | 0.9902 | 0.9927 | +0.0052 |
| flattened | 0 | 100 | 0.5717 | 0.6216 | 0.5787 | −0.0498 |
| flattened | 1 | 100 | 0.6281 | 0.6867 | 0.6411 | −0.0586 |
| flattened | 2 | 100 | 0.5995 | 0.6649 | 0.5888 | −0.0654 |
| inverted | 0 | 100 | 0.4567 | 0.4812 | 0.4842 | −0.0246 |
| inverted | 1 | 100 | 0.4975 | 0.5544 | 0.4968 | −0.0569 |
| inverted | 2 | 100 | 0.4287 | 0.5415 | 0.5147 | −0.1128 |

### 2.2 Aggregate across 3 seeds (gap only)

| Condition | Gap mean | Gap SD (across seeds) | \|mean\| > SD? |
|-----------|----------|------------------------|---------------|
| natural | +0.0041 | 0.0027 | Yes |
| flattened | −0.0580 | 0.0078 | Yes |
| inverted | −0.0647 | 0.0446 | Yes |

**Interpretation (cautious):** All three conditions pass the informal \|mean\| > SD test on the real−blank gap, but **natural's gap is ~0.4% on a ~99% scale** — distinguishable from zero only because seed SD is tiny; it is **not** evidence the model reads the image. Flattened/inverted show **blank confidence above real** by 2.5–11 pp per seed — the opposite of image-grounded behaviour.

Pooled 300-sample aggregates (matches `summary_hindi.json`):

| Condition | Real | Blank | Noise | Gap |
|-----------|------|-------|-------|-----|
| natural | 0.9940 | 0.9899 | 0.9904 | +0.0041 |
| flattened | 0.5998 | 0.6577 | 0.6029 | −0.0580 |
| inverted | 0.4610 | 0.5257 | 0.4986 | −0.0647 |

---

## 3. Probe 5 (calibration)

Tier 1/2 correctness scoring. Ten equal-width bins on [0, 1).

### 3.1 Per condition × seed — summary

| Condition | Seed | n | Accuracy | Mean conf | ECE |
|-----------|------|---|----------|-----------|-----|
| natural | 0 | 100 | 0.170 | 0.994 | 0.824 |
| natural | 1 | 100 | 0.140 | 0.993 | 0.853 |
| natural | 2 | 100 | 0.240 | 0.995 | 0.755 |
| flattened | 0 | 100 | 0.000 | 0.572 | 0.572 |
| flattened | 1 | 100 | 0.010 | 0.628 | 0.618 |
| flattened | 2 | 100 | 0.000 | 0.599 | 0.599 |
| inverted | 0 | 100 | 0.000 | 0.457 | 0.457 |
| inverted | 1 | 100 | 0.010 | 0.498 | 0.488 |
| inverted | 2 | 100 | 0.010 | 0.429 | 0.419 |

ECE = Σ (n_b/N) |acc_b − mean_conf_b| over occupied bins.

### 3.2 Per condition × seed — buckets (accuracy, n)

**natural seed0** — single bucket: [0.90, 1.00): n=100, acc=0.170  
**natural seed1** — [0.90, 1.00): n=97, acc=0.113  
**natural seed2** — [0.90, 1.00): n=98, acc=0.224  

**flattened seed0** — [0.40,0.50) n=15 acc=0.000; [0.50,0.60) n=59 acc=0.000; [0.60,0.70) n=19 acc=0.000; [0.70,0.80) n=4 acc=0.000; [0.80,0.90) n=2 acc=0.000; [0.90,1.00) n=1 acc=0.000  

**flattened seed1** — buckets n=2,47,34,8,3,6 across [0.40,1.00); only [0.90,1.00) acc=0.167 (n=6)  

**flattened seed2** — buckets n=10,50,28,9,1,2; all acc=0.000  

**inverted seed0** — seven buckets [0.30,1.00); all acc=0.000 (n=5 in [0.90,1.00))  
**inverted seed1** — seven buckets; [0.90,1.00) n=2 acc=0.500  
**inverted seed2** — seven buckets; [0.90,1.00) n=5 acc=0.200  

### 3.3 Pooled across 3 seeds (n=300 per condition)

| Condition | Accuracy | Mean conf | ECE | Dominant bucket |
|-----------|----------|-----------|-----|-----------------|
| natural | 0.183 | 0.994 | **0.811** | [0.90,1.00) n=295, acc=0.169 |
| flattened | 0.003 | 0.600 | **0.596** | [0.50,0.60) n=156, acc=0.000 |
| inverted | 0.007 | 0.461 | **0.454** | [0.40,0.50) n=148, acc=0.000 |

**Interpretation:** Natural shows **extreme overconfidence** — nearly all mass in the top decile with ~17% accuracy. Flattened/inverted spread confidence lower but **accuracy remains ~0%** in most buckets; calibration improves (lower ECE) only because confidence dropped, not because accuracy rose.

---

## 4. Cross-condition comparison (exposure dial)

| Metric | natural | flattened | inverted | Dial moved it? |
|--------|---------|-----------|----------|--------------|
| Mean confidence (real) | 0.994 | 0.600 | 0.461 | **Yes** (−0.39 / −0.53 vs natural) |
| Real−blank gap | +0.004 | −0.058 | −0.065 | **Yes** (sign flip) |
| Pooled accuracy | 0.183 | 0.003 | 0.007 | **Yes** (natural oddly highest) |
| Pooled ECE | 0.811 | 0.596 | 0.454 | **Yes** (lower but still miscalibrated) |

**What the numbers show:** Glyph-frequency exposure **strongly affects** reported confidence and calibration shape. The natural condition is where the headline finding lives (~99% confidence regardless of image content). Flattened/inverted are **not** “fixed” — they trade extreme overconfidence for near-zero accuracy with blank ≥ real confidence.

**What they would need to support a stronger claim:** That exposure *causes* better image grounding (not just lower logits). That would require positive real−blank gap at non-natural conditions or Probe 5b zero-script results — neither appears in this zip.

---

## 5. Training-curve probe (`probe3_curve_*.json`)

**Not present** in `Hindi Probe Results Final.zip`.

Cannot assess (a) structurally ungrounded confidence vs (b) undertraining from these files alone. Probe 3 natural results are consistent with (a) but do not rule out (b) without multi-step checkpoints.

---

## 6. Bengali archive (`Bengali Experiment Final.zip`)

Contents (not extracted into repo — checkpoints are ~235 MB × 9, gitignored):

| Path | Notes |
|------|-------|
| `checkpoints/checkpoint_{natural,flattened,inverted}_seed{0,1,2}.pt` | 9 checkpoints; **pre–script-scoped naming** (no `bengali_` prefix) |
| `checkpoints/tokenizer_{natural,flattened,inverted}.json` | 3 tokenizers; same legacy naming |
| `data/manifests/bengali_{natural,flattened,inverted}.jsonl` | Training manifests |
| `data/predictions/{paddleocr,surya,tesseract}/bengali.jsonl` | Stage 0 baselines |

**No probe3/probe5 jsonl for Bengali** in this zip.

---

## 7. Limitations stated plainly

- **n=3 seeds** per condition: fine for large condition shifts; weak for declaring small gaps “significant.”
- **n=100 crops/run:** bucket counts in Probe 5 are often sparse above 0.90 for non-natural conditions.
- **Natural accuracy ~18%** with ~99% confidence is the primary calibration failure; do not conflate with “model works slightly on natural.”
- Results reflect **one Colab training pass**; checkpoint naming on Drive may predate DECISIONS.md #47 script-scoped paths.

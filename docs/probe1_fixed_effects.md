# Probe 1 fixed-effects analysis

**Generated:** 2026-09-02  
**Script:** hindi  
**Inputs:** training manifests (`data/manifests/`), Probe 5 jsonl (`data/probe_results/`)  
**Method:** per-glyph accuracy ~ log(training exposure) + glyph fixed effects; one OLS fit per seed.

---

## 1. Feasibility pre-check (run before trusting any coefficient)

docs/results_analysis.md shows pooled line-level accuracy **18.3% (natural)**, **0.3% (flattened)**, **0.7% (inverted)**. A crossover FE fit needs usable accuracy variance in non-natural conditions — not just exposure variation.

### 1.1 Per-condition accuracy floor

| Condition | Probe5 line acc | n lines | Glyphs (≥3 eval tok) | Glyphs acc>0 | Frac acc>0 | Mean glyph acc | Frac acc=0 |
|-----------|-----------------|---------|----------------------|--------------|------------|----------------|------------|
| natural | 0.183 | 300 | 320 | 266 | 0.83 | 0.331 | 0.17 |
| flattened | 0.003 | 300 | 357 | 229 | 0.64 | 0.050 | 0.36 |
| inverted | 0.007 | 300 | 306 | 141 | 0.46 | 0.061 | 0.54 |

*Probe5 line acc* = Tier 1/2 whole-line correctness (matches `docs/results_analysis.md`).  
*Mean glyph acc* = per-cluster token accuracy from grapheme alignment (lenient when lines are wrong).

### 1.2 Verdict

**Headline exposure coefficient is NOT reported as meaningful.** Reasons:
- Flattened and inverted Probe 5 line accuracy both below 2% (flattened=0.3%, inverted=0.7%) — non-natural conditions did not learn readable OCR; FE fit would be dominated by floor effects.
- Flattened line accuracy 0.3% — too low for cross-condition exposure comparison.

A clear negative result: the current flattened/inverted runs collapsed to near-zero line accuracy before a within-glyph crossover could identify exposure effects.

---

## 2. Fixed-effects fit (per seed)

*Headline coefficient withheld — see §1.2.*

**Diagnostic fits (do not cite as headline):**
| Seed | n_obs | n_glyphs | β(log exposure) | SE | 95% CI | R² |
|------|-------|----------|-----------------|-----|--------|-----|
| 0 | 646 | 286 | +0.0424 | 0.0081 | [+0.0265, +0.0583] | 0.110 |
| 1 | 646 | 286 | +0.0465 | 0.0073 | [+0.0322, +0.0609] | 0.184 |
| 2 | 646 | 286 | +0.0398 | 0.0088 | [+0.0226, +0.0570] | 0.166 |

These positive β values are **not interpretable** as exposure effects: flattened/inverted line accuracy is ~0%, so the regression mostly compares natural-condition signal against near-zero floors with different exposure scales.

---

## 3. Why flattened/inverted accuracy collapsed

### 3.1 Training manifest text properties

| Condition | Lines | Unique glyphs | Glyph tokens | Entropy (bits) | Mean glyphs/line |
|-----------|-------|---------------|--------------|----------------|------------------|
| natural | 2538 | 351 | 89858 | 7.20 | 35.4 |
| flattened | 2707 | 594 | 87730 | 8.27 | 32.4 |
| inverted | 2872 | 469 | 87333 | 6.94 | 30.4 |

Flattened/inverted text is **synthesized** (DECISIONS.md #28) to hit target glyph PMFs — bigram-guided packing preserves some local structure but produces globally unnatural sentences. Higher unique-glyph count and different entropy vs natural are expected; the open question is whether the model failed to learn the visual task or learned a language prior that does not transfer to eval crops.

**Sample training line (first manifest row):**

- *natural:* `कहते हैं की वे अच्छी बावर्ची हैं। एक दिन एक दंत चिकित्सक के घर भयंकर`

- *flattened:* `विथी टूथवा लेथे ।उसल केसाअ निगके बार्सि लोगोंकोदी हचारमें मु वर्षबी`

- *inverted:* `कऊँ चेऔर कोप्रभा गढा लक्ष्मण झीतथासु दिन एशिया घोड़े केदण्डीकी कंहेगा।`

### 3.2 Final checkpoint training loss (if checkpoints available)

*No checkpoints found locally* (`checkpoints/checkpoint_{script}_{condition}_seed{N}.pt`). Compare final `loss` stored in Colab checkpoints: if flattened/inverted loss fell similarly to natural but accuracy stayed ~0%, the model optimized the training objective on unlearnable / prior-dominated text without acquiring readable OCR.

### 3.3 Confound vs exposure dial on confidence

docs/results_analysis.md shows mean confidence falling **0.99 → 0.60 → 0.46** across conditions. That pattern tracks **lower logits on synthesized text**, not necessarily successful exposure control with intact reading ability. With flattened/inverted accuracy at ~0%, the confidence drop **cannot** be interpreted as “the dial grounded the model” — it may reflect failure to learn the training distribution.

---

## 4. What this would need to become the headline number

1. **Non-natural conditions must produce >5–10% line-level accuracy** (or at least many glyphs with stable non-zero accuracy) so the FE fit is not a floor artifact.
2. **Longer training or milder inverted PMF** — current inverted synthesis may be too extreme for 5k steps at 19.5M params.
3. **Eval on held-out natural lines** while training on flattened/inverted (optional design change) to separate “can't read synthetic” from “exposure hurt this glyph”.
4. **Mixed-effects extension** (DECISIONS.md #46 / BOOK.md): random intercepts per seed; bootstrap over eval lines for intervals.
5. **Per-step loss curves** (`probe3_training_curve` + `--keep-snapshots`) to see whether natural/low-exposure conditions diverge during training.

---

## 5. Complexity (glyph fixed effects)

The fitted glyph dummies are the **complexity estimates** — not reported individually here (373 clusters). Export via `--dump-glyph-effects` (future) or refit with store of `beta[2:]` from the design matrix. Substantive interpretation requires a feasible fit (§1.2).

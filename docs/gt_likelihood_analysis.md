# GT-likelihood probe — teacher-forced log p(GT) and entropy

**Generated:** 2026-09-04  
**Code:** `src/probes/probe_gt_likelihood.py`  
**Inputs (committed):**
- `data/probe_results/probe_gt_likelihood_hindi_natural_seed0.jsonl`
- `data/probe_results/probe_gt_likelihood_hindi_natural_seed1.jsonl`
- `data/probe_results/probe_gt_likelihood_hindi_natural_seed2.jsonl`

**Sample:** Hindi/natural Tier C plain images; same `Random(0)` pool as
Probe 5b / Probe 6 real_plain (60 images). Per seed: **60 real + 60 blank
= 120 records**. Three seeds → **360 records total**. All numbers below
recomputed from those jsonl files (`mean_log_p_gt`, `mean_entropy`,
`step_log_p_gt`).

## Method
Teacher-force the **ground-truth** token sequence (not the model's own
generation) on each image. At every step record:
- `log p(gt_token)` under the full softmax,
- Shannon entropy `H(p)` of that full distribution (nats).

Length-normalized sequence score = mean of per-step `log p(gt)` over the
forced sequence (including `<EOS>`). Blank uses `make_blank` on the same
canonical-height crop. This estimator does **not** inherit max-softmax
self-selection bias (the decoder is scored on the GT id, not its argmax).

---

## Whole-sequence mean log p(GT)

| condition | n | mean log p(GT) | SD |
|---|---:|---:|---:|
| real | 180 | **−1.783** | 1.274 |
| blank | 180 | **−1.751** | 1.280 |

Exact recomputed means: real −1.7830151649, blank −1.7514361740.

Blank is **marginally higher** likelihood (less negative) than real — same
direction as several max-softmax blank ≥ real findings. The
real-vs-blank dissociation is therefore **not** an artifact of scoring
the decoder's own argmax.

## Mean predictive entropy

| condition | n | mean H(p) (nats) | SD |
|---|---:|---:|---:|
| real | 180 | **0.0210** | 0.0171 |
| blank | 180 | **0.0253** | 0.0170 |

Exact: real 0.0209560742, blank 0.0252738180.

Blank is marginally **higher** entropy (slightly less peaked), not lower.
Both sit near ~0.02 nats — the same order of magnitude — so entropy also
fails to separate real from blank in a way that would indicate
image-grounded uncertainty.

---

## First-token p(GT) — sharpest number

At step 0 there is **no correct linguistic prefix**; only the image
(or its absence) can help. Report `exp(mean of step_log_p_gt[0])` per
seed × condition:

| seed | real exp(mean log p) | blank exp(mean log p) |
|---:|---:|---:|
| 0 | **2.05×10⁻¹⁰** | **9.32×10⁻¹⁰** |
| 1 | **1.35×10⁻¹¹** | **3.13×10⁻¹¹** |
| 2 | **3.88×10⁻¹²** | **8.70×10⁻¹²** |

Exact `exp(mean_logp)`: seed0 real 2.051404e-10 / blank 9.315069e-10;
seed1 real 1.345207e-11 / blank 3.127641e-11; seed2 real 3.877068e-12 /
blank 8.697234e-12.

**Plain statement:** at the one decoding position with no correct
linguistic prefix available (the first token), p(GT) is catastrophically
low **and** statistically indistinguishable between real and blank images,
across all three seeds independently.

### Rest of sequence (tokens 2…)

Mean of per-record mean `step_log_p_gt[1:]`:

| condition | n | mean log p(GT) | ≈ exp(mean) |
|---|---:|---:|---:|
| real | 180 | **−1.140** | ~0.32 |
| blank | 180 | **−1.135** | ~0.32 |

Exact: real −1.1403309677, blank −1.1350623646.

From token 2 onward, p(GT) rises to ~0.32 **regardless of condition** —
present identically whether the forced prefix is being scored on a real
read or on a blank image.

---

## Mechanistic interpretation

This decomposes the earlier confidence-blindness finding into **where**
it originates:

1. At the position where **only the image could help** (no prefix yet),
   the model has **no measurable image-grounded signal** — first-token
   p(GT) is ~10⁻¹⁰–10⁻¹² on both real and blank.
2. At every later position, apparent fluency is **prefix-conditioned
   language modeling**, present identically under teacher forcing on
   blank as on real.

Estimator independence: the same real≈blank dissociation appears under
teacher-forced GT log-likelihood and entropy as under max-softmax over
self-generated tokens (DECISIONS.md #62).

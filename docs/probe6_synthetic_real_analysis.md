# Probe 6 — synthetic Claim B vs real Tier C (Hindi/natural, paper scope)

**Generated:** 2026-09-03

## Held-out validity (no leakage)
Before any inference, we assert that the Tier C real images under
`data/raw/hindi/images/` do not overlap with any training manifest image paths
under `data/manifests/hindi_{natural,flattened,inverted}.jsonl`.

Result (also recorded in `probe6_synthetic_real_gap.py` as a leakage guard):
- **0 overlaps**
- raw images checked: **120** files
- leakage-free flag in outputs: **all records have `leakage_free=true`**

## Scope (what this version does)
This reduced Probe 6 runs only on **Tier C real Hindi images** plus a **real-domain
blank control** on the same sample. It intentionally omits the full original
Probe 6 scope (Tier B degradation sweep, handwriting anecdote, held-out synthetic
pages 100–109, and multi-system gaps). Those remain future work.

## Pooled 3-seed results
Outputs:
- `data/probe_results/probe6_synthetic_real_hindi_seed0.jsonl`
- `data/probe_results/probe6_synthetic_real_hindi_seed1.jsonl`
- `data/probe_results/probe6_synthetic_real_hindi_seed2.jsonl`

Each seed has 60 GT rows; × 3 conditions (plain, degraded, blank) = **180** records per seed.
Pooled across 3 seeds: **540** records.

Correctness uses the same Tier 1/2 equivalence gate as Probe 5 calibration.
For the blank control, the json includes `correct: null`; for reporting we treat
`correct=null` as “not correct”, so `accuracy` is **0.0**.

| Condition | mean_confidence | accuracy |
|---|---:|---:|
| `real_plain` | **0.9861** | **0.0000** |
| `real_degraded` | **0.9768** | **0.0000** |
| `blank` | **0.9799** | **0.0000** |

## Finding
The **confidence-blindness pattern replicates** on real Tier C documents:
real (plain/degraded) mean confidence stays near the blank-control level,
while Tier 1/2 line accuracy is **0.0** across the same real sample.

So the Claim B mechanistic interpretation (“confidence does not track image
readability”) does not depend on the synthetic renderer domain.

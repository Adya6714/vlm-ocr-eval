# Probe 2 — confusion structure (GT-aligned, p(true) + rank)

**Generated:** 2026-09-03

## Inputs
Each `probe2_hindi_natural_seed{N}.jsonl` file is a **single JSON object** with
an `edges` list (not line-delimited JSONL), so analysis uses `json.load()`.

Files:
- `data/probe_results/probe2_hindi_natural_seed0.jsonl`
- `data/probe_results/probe2_hindi_natural_seed1.jsonl`
- `data/probe_results/probe2_hindi_natural_seed2.jsonl`

## Boundary-noise metric
Edge objects carry `weight` and two endpoints:
- `chosen` (the model’s argmax at a step)
- `confused_with` (which runner-up mass the model nearly picked)

To quantify boundary-token noise (where the model emits `space` / `<EOS>`-like
clusters), we define boundary tokens as:
- `<EOS>`
- `' '` (space)

We report the fraction of **total edge weight** that touches boundary tokens via
`confused_with`:

- **boundary edge-weight fraction** = **0.0891** (8.9%)

## Top confusion pairs after excluding boundary destination edges
We exclude any edge where `confused_with` is a boundary token, then sort remaining
edges by `weight` and take the top 10.

Top 10 pairs (token_i → token_j):
1. `दु` → `औ` (w=0.7907)
2. `पा` → `फि` (w=0.6418)
3. `क` → `दो` (w=0.6028)
4. `र्व` → `री` (w=0.5978)
5. `उ` → `य` (w=0.5231)
6. `' '` → `चु` (w=0.5096)
7. `भा` → `मू` (w=0.4907)
8. `र` → `सा` (w=0.4895)
9. `म` → `क` (w=0.4795)
10. `ता` → `न` (w=0.4690)

## Qualitative read (visually/phonetic clustering)
Across these top-weight confusion pairs (after boundary exclusion), the simple
Unicode-similarity heuristic labels (e.g. “adjacent codepoint” vs “dissimilar”)
are mixed rather than forming a tight cluster around a small set of matra-family
neighbors. This supports the paper’s Claim B story: remaining “confusions” are
not dominated by consistent local visual/phonetic near-neighbors.

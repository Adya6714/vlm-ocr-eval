# Bootstrap CIs on ANOVA variance shares (Step 6)

**Method:** 2000 replicates; resample 60 image rows with replacement;
keep all three seeds for each drawn row; `anova_var_decomp` as in
`src/analysis/paper_defensibility_stats.py`. RNG seed 0.
Point shares are the original Follow-Up 5 numbers (same function, full panel).

| Panel | Image % [95% CI] | Seed % [95% CI] | Residual % [95% CI] |
|---|---|---|---|
| Probe 5b Hindi Confidence | 9.1 [0.0, 22.4] | 7.1 [0.2, 17.7] | 83.9 [67.9, 96.6] |
| Probe 5b Blank Confidence | 0.0 [0.0, 0.0] | 18.0 [11.0, 27.7] | 82.0 [72.3, 89.0] |
| GT Likelihood (real) | 84.1 [69.7, 91.1] | 1.5 [0.4, 4.3] | 14.4 [8.0, 27.4] |
| GT Likelihood (blank) | 85.0 [70.8, 90.9] | 2.2 [0.7, 5.9] | 12.8 [7.4, 25.5] |

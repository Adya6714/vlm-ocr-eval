# Tier 2 — RLVR coverage ablation: not completed

**What this is not:** a coverage-term ablation. The design from BOOK.md
Chapter 6 requires retraining from the SFT checkpoint with the coverage
reward term removed (λ=0), then checking whether the retrained policy
omits difficult content relative to the SFT baseline. No retraining
occurred.

**What actually ran:** the existing SFT checkpoint's own greedy outputs,
scored once against the λ=0 reward function, with no policy update.
n=32, mean_coverage=0.0 across all 32 examples, all below the 0.5
threshold. This says the plain SFT model (100 training steps) doesn't
cover its inputs well on its own — plausibly just an undertrained model
after a short smoke-test SFT run — and says nothing about whether
removing the coverage reward causes omission, since coverage was never
part of a training signal here.

**Why retraining didn't happen:** it was never in the executed script.
`src/models/demo/rlvr_train.py` prints that full PPO is not implemented
and, when an SFT adapter exists, greedy-decodes that adapter and writes
`rlvr_summary.json` instead of a policy update. That is also the note
inside the summary JSON. This session did not crash out of an RL loop
and did not run out of time on one: the loop was not attempted. Cell 10
stdout was not kept in the paste, so a per-example `generate()` crash
versus empty or useless decode is not established; only the scored
summary (n=32, mean_coverage=0.0) is.

Not pursued further this session. SFT itself (Decision #84 closing #3,
real trained adapter on `ds4sd/SmolDocling-256M-preview`) is confirmed
working and is the result this session actually produced.

Raw summary: Colab wrote `checkpoints/demo_rlvr_nocov/rlvr_summary.json`
(not in this git tree). Reward-shape tests remain in
`tests/test_demo_rlvr_order.py`; those are not a trained-policy result.

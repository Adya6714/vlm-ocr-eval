⏳ Probe 5b — still not written, still optional, still lowest priority
✅ README.md — done, committed
🔄 Website — base version built and shared; Cursor is now extending it (Repository Tour, Paper-mode toggle, Talking Points) per the prompt just given — waiting on Cursor's report back
✅ IMPLEMENTATION.md cleanup — Probe 2 filename fixed, done
⏳ BOOK.md — still untouched, still pending, still lowest priority
🔄 TODO.md — got the deferred-stages note, but hasn't been updated to reflect the make smoke-test bug fixes (commit 645ae11 happened after the TODO.md edit) — small follow-up still open
✅ DECISIONS.md — done, both additions landed: #45 (Probe 6 resize decision) and the #6 verification append (Sarvam numbers confirmed current)

New, not on the original list:

✅ make smoke-test bugs — fixed, verified passing clean, this is now a real live demo asset

Blocked on Colab (status unknown until you check):

🔴 Probe 1 real numbers — training was mid-inverted seed0 at last check; likely finished by now, but probing (probe_all.sh) was queued after training and hasn't been confirmed run
🔴 Probe 5 calibration table, aggregated — same dependency as above
🔴 Bengali full sweep — queued after Hindi's probing, status unknown
🔴 Probe 6, instrument half — code ready, still needs the held-out pages (100–109) rendered and predictions generated
🔴 Probe 6, baseline half — still genuinely unknown whether run_baselines.py works against the synthetic folder layout without changes

Deliberately out of scope, unchanged, now documented everywhere:

❌ Stage 2b — LoRA demo model
❌ Stage 3 — reading-order metrics
❌ Stage 5 — Sarvam API transfer
❌ Stage 6 — cascade triage
❌ Probe 4 — already covered by Stage 0

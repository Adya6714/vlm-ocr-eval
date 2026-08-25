# aksharaprobe — repo-wide runner
#
# Two intended uses, matching this project's two-story presentation:
#   `make smoke-test`  — proves the ENTIRE pipeline runs end-to-end on
#                         tiny fake data, in seconds, no GPU needed.
#                         This is the "here's proof the architecture
#                         works" story -- usable even with zero real
#                         compute or results.
#   `make stage0` etc. — the real pipeline, against real data. Needs
#                         the data/ files this repo doesn't ship with
#                         (see README.md) and, for Stage 2+, a GPU
#                         (Colab T4 -- see AGENTS.md's Colab convention).
#
# Targets are only added here once a script is actually built and
# tested per IMPLEMENTATION.md -- an empty/placeholder target would
# lie about what's runnable, which defeats the point of this file.

.PHONY: help smoke-test stage0-fetch stage0-baselines stage0-tier-tests \
        stage0-hand-review stage0-taxonomy stage2-instrument-smoke \
        probe1-smoke clean-smoke

help:
	@echo "aksharaprobe — available targets:"
	@echo "  make smoke-test          — run everything runnable on fake data, prove the pipeline works"
	@echo "  make stage0-fetch        — pull GlotOCR-bench ground truth (real, needs HF auth)"
	@echo "  make stage0-baselines    — run Tesseract/Surya/PaddleOCR over data/raw (real, slow -- prefer Colab)"
	@echo "  make stage0-tier-tests   — run Tier 1 + Tier 2 equivalence self-tests"
	@echo "  make stage0-taxonomy     — produce the Stage 0 summary report (data/predictions/error_taxonomy.csv)"
	@echo "  make stage2-instrument-smoke — tokenizer/encoder/decoder/generate smoke tests, fake data"
	@echo "  make probe1-smoke        — run all 9 Probe 1 training runs on fake data (proves orchestration works)"
	@echo "  make clean-smoke         — remove smoke-test artifacts from /tmp"

# --- Stage 0: real pipeline ---

stage0-fetch:
	python3 src/data_pipeline/fetch_glotocr.py

stage0-baselines:
	@echo "NOTE: this is slow and CPU-heavy locally -- prefer running on"
	@echo "Colab per AGENTS.md's Colab convention. See README.md for the"
	@echo "upload/export steps."
	PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True python3 src/eval/run_baselines.py \
		--engine tesseract --engine surya --engine paddleocr \
		--per-image-timeout-seconds 30

stage0-tier-tests:
	python3 src/eval/equivalence_tables.py
	python3 src/eval/transliteration_equivalence.py

stage0-hand-review-hindi:
	PYTHONPATH=src/eval python3 src/eval/hand_review.py hindi

stage0-hand-review-bengali:
	PYTHONPATH=src/eval python3 src/eval/hand_review.py bengali

stage0-taxonomy:
	python3 src/eval/error_taxonomy.py

# --- Stage 2a: instrument model, smoke-testable without real data ---

stage2-instrument-smoke:
	cd src/models/instrument && python3 tokenizer.py
	cd src/models/instrument && python3 encoder.py
	cd src/models/instrument && python3 decoder.py

probe1-smoke:
	@echo "Running all 9 Probe 1 runs (3 conditions x 3 seeds) on FAKE"
	@echo "data -- this proves the training + resumability orchestration"
	@echo "works end to end. It does NOT produce a real finding -- fake"
	@echo "data carries no real exposure signal to measure."
	python3 scripts/make_fake_probe1_data.py
	python3 src/probes/probe1_exposure.py \
		--natural-manifest /tmp/fake_lines/natural.jsonl \
		--flattened-manifest /tmp/fake_lines/flattened.jsonl \
		--inverted-manifest /tmp/fake_lines/inverted.jsonl \
		--output-root /tmp/probe1_smoke \
		--total-steps 10 --batch-size 4 --log-every 5 --checkpoint-every 10
	cd src/models/instrument && python3 generate.py

# --- Everything smoke-testable, in one command ---

smoke-test: stage0-tier-tests stage2-instrument-smoke probe1-smoke
	@echo ""
	@echo "=== SMOKE TEST COMPLETE ==="
	@echo "Every stage that can run without real data or a GPU just ran"
	@echo "end to end: tokenizer, encoder, decoder, generation, and all"
	@echo "9 Probe 1 training + resumability runs. This proves the"
	@echo "architecture and orchestration are real and working, even"
	@echo "with zero real training data or compute behind it."

clean-smoke:
	rm -rf /tmp/fake_lines /tmp/probe1_smoke /tmp/probe1_test /tmp/test_ckpt

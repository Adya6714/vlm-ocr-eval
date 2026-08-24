"""Checkable properties of run_baselines Colab paths and resume."""

import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))

import run_baselines  # noqa: E402


def _fake_engine_always_ok(image_path, language):
    """
    Module-level OCR fake for multiprocessing tests.

    Why this exists: with process-level isolation in run_baselines, engine
    callables must be pickleable under the multiprocessing `spawn` method.
    """
    return {"predicted_text": "ok", "confidence": 1.0, "skipped_reason": None}


def _fake_engine_raise_if_called(image_path, language):
    """
    Module-level OCR fake that fails if invoked.

    Why this exists: we want a resume-skip test that would fail if
    `run_engine_over_language()` accidentally re-ran already-completed images.
    """
    raise AssertionError("engine_fn should not have been called")


def _fake_engine_sleep_on_0(image_path, language):
    """
    Simulate a runaway backend by sleeping on the first image only."""
    if os.path.basename(image_path) == "0_plain.png":
        time.sleep(2)
    return {"predicted_text": "ok", "confidence": 1.0, "skipped_reason": None}


def _fake_engine_threadpool_shutdown_hang_on_0(image_path, language):
    """
    Simulate a backend hang in `ThreadPoolExecutor.shutdown(wait=True)`.

    Why this exists: SIGALRM-based timeouts do not reliably interrupt some
    thread/join/lock-acquire hang patterns; process-level killing must work.
    """
    if os.path.basename(image_path) != "0_plain.png":
        return {"predicted_text": "ok", "confidence": 1.0, "skipped_reason": None}

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    lock.acquire()  # Held forever, so worker thread blocks on acquire.

    ex = ThreadPoolExecutor(max_workers=1)
    ex.submit(lock.acquire)  # Worker blocks on acquiring `lock`.
    ex.shutdown(wait=True)  # Hang: wait forever for that worker.


class ResolveDataPathsTests(unittest.TestCase):
    def test_default_is_repo_data_layout(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCR_DATA_ROOT", None)
            os.environ.pop("OCR_PRED_ROOT", None)
            root, raw, pred = run_baselines.resolve_data_paths()
        self.assertEqual(root, "data")
        self.assertEqual(raw, os.path.join("data", "raw"))
        self.assertEqual(pred, os.path.join("data", "predictions"))

    def test_one_root_moves_both_input_and_output(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCR_PRED_ROOT", None)
            root, raw, pred = run_baselines.resolve_data_paths("/content/data")
        self.assertEqual(raw, os.path.join("/content/data", "raw"))
        self.assertEqual(pred, os.path.join("/content/data", "predictions"))

    def test_pred_root_override_does_not_move_raw(self):
        root, raw, pred = run_baselines.resolve_data_paths(
            "/content/data", pred_root="/tmp/smoke-preds"
        )
        self.assertEqual(raw, os.path.join("/content/data", "raw"))
        self.assertEqual(pred, "/tmp/smoke-preds")


class CanonicalPathTests(unittest.TestCase):
    def test_colab_absolute_path_becomes_repo_relative(self):
        path = run_baselines.canonical_image_path(
            "hindi", "/content/data/raw/hindi/images/23_plain.png"
        )
        self.assertEqual(path, "data/raw/hindi/images/23_plain.png")


class ResumeTests(unittest.TestCase):
    def test_skips_completed_and_does_not_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw" / "hindi" / "images"
            pred = Path(tmp) / "predictions"
            raw.mkdir(parents=True)
            for name in ("0_plain.png", "1_plain.png"):
                (raw / name).write_bytes(b"\x89PNG\r\n\x1a\n")

            run_baselines.run_engine_over_language(
                "tesseract",
                _fake_engine_always_ok,
                "hindi",
                variant="plain",
                raw_root=str(Path(tmp) / "raw"),
                pred_root=str(pred),
            )

            out = pred / "tesseract" / "hindi.jsonl"
            before = out.read_text(encoding="utf-8")

            run_baselines.run_engine_over_language(
                "tesseract",
                _fake_engine_raise_if_called,
                "hindi",
                variant="plain",
                raw_root=str(Path(tmp) / "raw"),
                pred_root=str(pred),
            )

            after = out.read_text(encoding="utf-8")
            self.assertEqual(before, after)
            rows = [json.loads(line) for line in after.splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0]["image_path"], "data/raw/hindi/images/0_plain.png"
            )

    def test_per_image_timeout_marks_skipped_reason_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw" / "hindi" / "images"
            pred = Path(tmp) / "predictions"
            raw.mkdir(parents=True)
            for name in ("0_plain.png", "1_plain.png"):
                (raw / name).write_bytes(b"\x89PNG\r\n\x1a\n")

            run_baselines.run_engine_over_language(
                "tesseract",
                _fake_engine_sleep_on_0,
                "hindi",
                variant="plain",
                raw_root=str(Path(tmp) / "raw"),
                pred_root=str(pred),
                per_image_timeout_seconds=1.0,
            )

            out = pred / "tesseract" / "hindi.jsonl"
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 2)

            by_id = {r["id"]: r for r in rows}
            self.assertEqual(by_id["0"]["skipped_reason"], "timeout")
            self.assertIsNone(by_id["1"]["skipped_reason"])

    def test_process_timeout_kills_threadpool_shutdown_hang(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw" / "hindi" / "images"
            pred = Path(tmp) / "predictions"
            raw.mkdir(parents=True)
            for name in ("0_plain.png", "1_plain.png"):
                (raw / name).write_bytes(b"\x89PNG\r\n\x1a\n")

            t0 = time.perf_counter()
            run_baselines.run_engine_over_language(
                "tesseract",
                _fake_engine_threadpool_shutdown_hang_on_0,
                "hindi",
                variant="plain",
                raw_root=str(Path(tmp) / "raw"),
                pred_root=str(pred),
                per_image_timeout_seconds=1.0,
            )
            elapsed = time.perf_counter() - t0

            # Spawn overhead can be noticeable on macOS CI, but a real hang
            # should still be forced down within seconds (not minutes).
            self.assertLess(elapsed, 6.0)

            out = pred / "tesseract" / "hindi.jsonl"
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 2)

            by_id = {r["id"]: r for r in rows}
            self.assertEqual(by_id["0"]["skipped_reason"], "timeout")
            self.assertIsNone(by_id["1"]["skipped_reason"])

    def test_incomplete_trailing_line_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hindi.jsonl"
            complete = {
                "id": "1",
                "variant": "plain",
                "engine": "tesseract",
                "language": "hindi",
            }
            path.write_text(
                json.dumps(complete) + "\n" + '{"id": "2", "variant": "plain"',
                encoding="utf-8",
            )
            done = run_baselines.load_completed_keys(str(path))
            self.assertEqual(
                done, {("tesseract", "hindi", "1", "plain")}
            )
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                json.dumps(complete) + "\n",
            )


class ExportZipTests(unittest.TestCase):
    def test_members_are_implementation_stage0_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predictions"
            engine_dir = pred / "surya"
            engine_dir.mkdir(parents=True)
            (engine_dir / "hindi.jsonl").write_text(
                '{"id":"0","engine":"surya","language":"hindi"}\n',
                encoding="utf-8",
            )
            zip_path = Path(tmp) / "predictions.zip"
            written = run_baselines.export_predictions_zip(str(pred), str(zip_path))
            self.assertEqual(written, ["data/predictions/surya/hindi.jsonl"])
            with zipfile.ZipFile(zip_path) as zf:
                self.assertEqual(
                    zf.namelist(), ["data/predictions/surya/hindi.jsonl"]
                )


if __name__ == "__main__":
    unittest.main()

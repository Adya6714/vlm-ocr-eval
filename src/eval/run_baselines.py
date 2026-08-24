"""
Runs Tesseract, Surya, and PaddleOCR over every image in
{data_root}/raw/{language}/images/ and saves raw predictions.

Why this exists: Stage 0's error taxonomy needs (ground_truth, predicted)
pairs per engine. This script produces the "predicted" half — it does
not judge correctness, that's error_taxonomy.py's job later.

This is a CPU/GPU-heavy batch job (AGENTS.md "Heavy scripts run on
Colab"). One configurable root (`--data-root` / `OCR_DATA_ROOT`,
default `data/`) holds both inputs (`{root}/raw`) and outputs
(`{root}/predictions`). `OCR_PRED_ROOT` / `--pred-root` is a nested
override so a resume smoke test can write somewhere other than a live
`data/predictions/` tree without editing this file.

Not every engine supports every script well. Tesseract has no real
Santhali/Kashmiri language pack; PaddleOCR's multilingual coverage for
Ol Chiki and Perso-Arabic-Kashmiri specifically is thin to nonexistent.
Rather than skip those combinations silently, this script still runs
them and records whatever comes out (including empty/garbage output) —
a model confidently producing garbage on a script it doesn't support is
itself a data point (this connects to GlotOCR Bench's own finding that
models don't fail silently, they hallucinate fluent-looking text in a
script they DO know — worth watching for here too).

Output layout (IMPLEMENTATION.md Stage 0 — same on Colab as locally):
    data/predictions/{engine}/{language}.jsonl
    one line per image:
        {id, variant ("plain"/"degraded"), predicted_text, confidence,
         image_path, engine, language}

`image_path` is always the repo-canonical `data/raw/...` path, not a
Colab absolute path, so the jsonl is usable after export.

Safe to re-run: appends and skips (engine, language, id, variant)
already present in the output file (AGENTS.md Long-running scripts).
To redo from scratch, delete the jsonl first.

Colab last cell (export back into this repo):
    python src/eval/run_baselines.py --data-root /content/data \\
        --export-only --export-zip /content/predictions.zip
    from google.colab import files
    files.download("/content/predictions.zip")
    # unzip at the repo root → data/predictions/{engine}/{language}.jsonl
"""

import argparse
import glob
import json
import os
import multiprocessing
import queue as _queue
import time
import zipfile

import pytesseract
from PIL import Image

# Tesseract language-pack codes differ from GlotOCR's ISO codes.
# Only languages Tesseract actually ships a trained model for are
# listed here; anything else is skipped for Tesseract specifically
# (but still attempted for Surya/PaddleOCR below).
TESSERACT_LANG_MAP = {
    "hindi": "hin",
    "bengali": "ben",
    # santhali, kashmiri: no standard Tesseract traineddata exists.
    # Intentionally omitted rather than guessed — a wrong lang code
    # would silently produce garbage attributed to the wrong cause.
}

# IMPLEMENTATION.md Stage 0's output path, used as zip member names so
# a Colab download unpacks into this repo without a remap.
REPO_PRED_PREFIX = os.path.join("data", "predictions")

LANGUAGES = ["hindi", "bengali", "santhali", "kashmiri"]


def resolve_data_paths(
    data_root: str | None = None,
    pred_root: str | None = None,
) -> tuple[str, str, str]:
    """
    Single Colab/local knob: data_root holds both inputs and outputs.

    Returns (data_root, raw_root, pred_root). Defaults match this repo's
    `data/` layout. `--pred-root` / `OCR_PRED_ROOT` is a nested override
    so a resume smoke test can write somewhere other than a live
    predictions tree (DECISIONS.md #31) — not a second Colab path.

    Called from: main(), after argparse; tests call it directly so they
    never have to patch module-level globals.
    """
    root = (
        data_root
        if data_root is not None
        else os.environ.get("OCR_DATA_ROOT", "data")
    )
    raw = os.path.join(root, "raw")
    if pred_root is not None:
        pred = pred_root
    elif os.environ.get("OCR_PRED_ROOT"):
        pred = os.environ["OCR_PRED_ROOT"]
    else:
        pred = os.path.join(root, "predictions")
    return root, raw, pred


# Import-time defaults so helper functions work when called without an
# explicit root (local `python src/eval/run_baselines.py` with cwd =
# repo). main() re-resolves after argparse so Colab `--data-root` wins.
DATA_ROOT, RAW_ROOT, PRED_ROOT = resolve_data_paths()


def canonical_image_path(language: str, image_path: str) -> str:
    """
    Repo-relative image path written into the predictions jsonl.

    Always `data/raw/{language}/images/{filename}`, never the Colab
    absolute path. hand_review.py prints this; after the export zip
    lands in this repo, it has to resolve against the checkout, not
    against a VM that no longer exists.
    """
    return (
        f"data/raw/{language}/images/{os.path.basename(image_path)}"
    )


def load_ground_truth_index(language: str, *, raw_root: str | None = None) -> dict:
    """
    Reads ground_truth.jsonl for one language and ins it by id, so
    later stages can look up an image's ground truth without
    re-parsing the file. Not used for scoring here — run_baselines.py
    only produces predictions — but returned for convenience since
    callers often want both together.
    """
    root = raw_root if raw_root is not None else RAW_ROOT
    gt_path = os.path.join(root, language, "ground_truth.jsonl")
    index = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["id"]] = row
    return index


def iter_images(
    language: str,
    variant: str | None = None,
    *,
    raw_root: str | None = None,
):
    """
    Yields (image_id, variant, image_path) for every plain/degraded
    image saved under {raw_root}/{language}/images/. Variant is "plain"
    or "degraded", parsed from the filename fetch_glotocr.py wrote.

    If variant is set, only that variant is yielded (e.g. plain-only
    smoke runs without touching degraded copies).
    """
    root = raw_root if raw_root is not None else RAW_ROOT
    img_dir = os.path.join(root, language, "images")
    for path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
        filename = os.path.basename(path)
        stem = filename.rsplit(".", 1)[0]  # e.g. "23_plain"
        image_id, image_variant = stem.rsplit("_", 1)
        if variant is not None and image_variant != variant:
            continue
        yield image_id, image_variant, path


def load_completed_keys(out_path: str) -> set[tuple[str, str, str, str]]:
    """
    Reads an existing predictions jsonl and returns the set of
    (engine, language, image_id, variant) already written.

    Why this exists: AGENTS.md requires batch scripts to resume by
    default. run_engine_over_language calls this on startup so a killed
    or crashed run can be restarted without redoing finished images or
    wiping the partial file (the previous open(..., "w") behaviour).

    Incomplete trailing lines (possible if killed mid-write) are ignored
    for the done-set and truncated from the file so consumers never see
    a half-written JSON object. Within one {engine}/{language}.jsonl the
    identity that matters is (id, variant); engine/language are kept in
    the key so the same helper stays correct if outputs are ever merged.
    """
    done: set[tuple[str, str, str, str]] = set()
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return done

    good_end = 0  # byte offset after the last fully-parsed line
    with open(out_path, "rb") as f:
        while True:
            line_start = f.tell()
            raw = f.readline()
            if not raw:
                break
            try:
                row = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Truncate from this incomplete/corrupt line onward.
                good_end = line_start
                break
            done.add((row["engine"], row["language"], row["id"], row["variant"]))
            good_end = f.tell()

    file_size = os.path.getsize(out_path)
    if good_end < file_size:
        with open(out_path, "rb+") as f:
            f.truncate(good_end)

    return done


def _engine_process_runner(engine_fn, image_path: str, language: str, out_queue):
    """
    Child-process entrypoint for one OCR engine invocation.

    Why this exists: `run_engine_with_timeout()` must be able to terminate a
    backend even if it hangs in a way that ignores SIGALRM (e.g. a
    `ThreadPoolExecutor.shutdown() -> thread.join() -> lock.acquire()` chain).
    A separate process lets us kill the whole OS process reliably.

    Called from: `run_engine_with_timeout()` once per image, per engine.
    """
    try:
        result = engine_fn(image_path, language)
        out_queue.put(("ok", result))
    except Exception as e:  # noqa: BLE001 - forward error as string
        out_queue.put(("exc", repr(e)))


def run_engine_with_timeout(
    engine_fn,
    image_path: str,
    language: str,
    *,
    timeout_seconds: float,
) -> dict:
    """
    Calls engine_fn(image_path, language) with a hard per-image timeout.

    Why this exists: one bad/out-of-distribution image can make an OCR
    backend run indefinitely, including in hang patterns that SIGALRM cannot
    reliably interrupt (e.g. a ThreadPoolExecutor shutdown/join/lock-acquire
    chain). We therefore isolate each call in a separate OS process and
    enforce the timeout by terminating the process.

    Called from: run_engine_over_language(), once per image, per engine.
    """
    if timeout_seconds is None:
        return engine_fn(image_path, language)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0 when set")

    ctx = multiprocessing.get_context("spawn")
    out_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_engine_process_runner,
        args=(engine_fn, image_path, language, out_queue),
    )

    started = False
    timed_out = False
    try:
        proc.start()
        started = True
        proc.join(timeout_seconds)
        timed_out = proc.is_alive()

        if timed_out:
            # Best-effort cleanup: terminate first, then force-kill if needed.
            # Killing guarantees termination even if the process is blocked
            # mid-thread-join.
            proc.terminate()
            proc.join(1.0)
            if proc.is_alive():
                try:
                    proc.kill()
                except AttributeError:
                    proc.terminate()
                proc.join(1.0)

        try:
            status, payload = out_queue.get_nowait()
        except _queue.Empty:
            status, payload = None, None

        if status == "ok":
            return payload
        if status == "exc":
            raise RuntimeError(payload)
        if timed_out:
            return {
                "predicted_text": None,
                "confidence": None,
                "skipped_reason": "timeout",
            }
        raise RuntimeError(
            "engine process exited without result "
            f"(exit_code={proc.exitcode})"
        )
    finally:
        # Avoid leaking a child process if something unexpected happens.
        if started and proc.is_alive():
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                proc.terminate()
        try:
            out_queue.close()
            out_queue.join_thread()
        except Exception:  # noqa: BLE001
            pass


# Lazy singletons — model load is seconds per engine; constructing these
# inside run_surya/run_paddleocr per image made debugging runs unusable.
_SURYA_REC = None
_PADDLE_BY_LANG: dict[str, object] = {}


def _surya_page_text(page) -> str:
    """
    Flatten one Surya 0.22 PageOCRResult into plain text.

    Full-page OCR returns HTML per layout block; BeautifulSoup strips
    tags while preserving reading order via block.reading_order.
    """
    from bs4 import BeautifulSoup

    lines = []
    for blk in sorted(page.blocks, key=lambda b: b.reading_order):
        if blk.skipped or blk.error or not blk.html:
            continue
        text = BeautifulSoup(blk.html, "html.parser").get_text(
            separator=" ", strip=True
        )
        if text:
            lines.append(text)
    return "\n".join(lines)


def _get_surya_recognizer():
    """One RecognitionPredictor per process (surya-ocr 0.22 API)."""
    global _SURYA_REC
    if _SURYA_REC is None:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor

        _SURYA_REC = RecognitionPredictor(SuryaInferenceManager())
    return _SURYA_REC


def run_tesseract(image_path: str, language: str) -> dict:
    """
    Runs Tesseract on one image. Returns predicted text and, where
    Tesseract exposes it, a mean word-level confidence (0-100, not
    0-1 — normalize later at analysis time, not here, to keep this
    function a thin wrapper rather than a place decisions get buried).

    Skipped languages return a clearly-flagged None result rather than
    a fabricated empty string, so downstream code can tell "engine
    doesn't support this" apart from "engine tried and produced
    nothing."
    """
    tess_lang = TESSERACT_LANG_MAP.get(language)
    if tess_lang is None:
        return {"predicted_text": None, "confidence": None,
                "skipped_reason": "no tesseract traineddata for this language"}

    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=tess_lang)

    # image_to_data gives per-word confidences; average the non-negative
    # ones (Tesract returns -1 for words it has no confidence for).
    data = pytesseract.image_to_data(image, lang=tess_lang,
                                      output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
    mean_conf = sum(confidences) / len(confidences) if confidences else None

    return {"predicted_text": text.strip(), "confidence": mean_conf,
            "skipped_reason": None}


def run_surya(image_path: str, language: str) -> dict:
    """
    Runs Surya OCR on one image. Surya's recognition is largely
    language-agnostic (it detects script from the image rather than
    needing a language hint the way Tesseract does), so unlike
    run_tesseract this is attempted for every language, including
    Santhali and Kashmiri.

    Import is local to the lazy singleton, not module-level, because
    surya's model loading is slow and only needed if this function is
    actually called — keeps a Tesseract-only debugging run fast.

    surya-ocr 0.22+: full-page RecognitionPredictor (typo fix:
    surya.recognition, not surya.rognition). Requires llama-server
    on PATH (brew install llama.cpp) — see SuryaInferenceManager docs.
    """
    rec_predictor = _get_surya_recognizer()
    image = Image.open(image_path).convert("RGB")
    pages = rec_predictor([image], full_page=True)
    full_text = _surya_page_text(pages[0])

    # Block-level confidence is not exposed in 0.22's BlockOCRResult;
    # leave confidence None rather than invent a number.
    return {"predicted_text": full_text.strip(), "confidence": None,
            "skipped_reason": None}


def _get_paddle_ocr(paddle_lang: str):
    """
    One PaddleOCR pipeline per lang code (PaddleOCR 3.x API).

    PaddleOCR 3.7 rejects the old 2.x constructor kwargs: `show_log`
    and `use_angle_cls` raise ValueError("Unknown argument: ...") via
    paddleocr._common_args.parse_common_args. Use the 3.x names only:
    use_textline_orientation (was use_angle_cls), predict() (was ocr()).
    Hindi uses lang='hi' (Devanagari rec bundle), not the old
    'devanagari' alias.

    enable_mkldnn=False: on some CPU wheels (incl. local macOS and
    Colab CPU fallback) MKLDNN/OneDNN hits
    ConvertPirAttribute2RuntimeAttribute NotImplementedError mid-batch.
    Disabling MKLDNN keeps the same predict()/jsonl protocol; it only
    changes the CPU inference backend. GPU paths ignore MKLDNN anyway.
    """
    from paddleocr import PaddleOCR

    if paddle_lang not in _PADDLE_BY_LANG:
        _PADDLE_BY_LANG[paddle_lang] = PaddleOCR(
            lang=paddle_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _PADDLE_BY_LANG[paddle_lang]


def run_paddleocr(image_path: str, language: str) -> dict:
    """
    Runs PaddleOCR on one image. PaddleOCR's multilingual model
    selection is coarser than Tesseract's — it groups many languages
    under a handful of model bundles ('hi', 'en', 'ar', etc.) rather
    than one model per language, so the mapping below is approximate.
    Attempted for all four languages; expect weak-to-garbage output on
    Santhali/Kashmiri specifically, which is itself worth recording,
    not hiding.
    """
    # PaddleOCR 3.x lang codes — see paddleocr._utils.langs.
    PADDLE_LANG_MAP = {
        "hindi": "hi",
        "bengali": "en",  # no dedicated Bengali bundle in 3.7; en is the
                           # documented fallback and expected to perform
                           # poorly — that gap is itself a taxonomy point.
        "santhali": "en",
        "kashmiri": "ar",  # Arabic-script model, closest to Perso-Arabic.
    }
    paddle_lang = PADDLE_LANG_MAP.get(language, "en")

    ocr = _get_paddle_ocr(paddle_lang)
    results = ocr.predict(image_path)

    if not results:
        return {"predicted_text": "", "confidence": None, "skipped_reason": None}

    page = results[0]
    lines = page.get("rec_texts") or []
    confs = page.get("rec_scores") or []
    if not lines:
        return {"predicted_text": "", "confidence": None, "skipped_reason": None}

    full_text = "\n".join(lines)
    mean_conf = sum(confs) / len(confs) if confs else None

    return {"predicted_text": full_text.strip(), "confidence": mean_conf,
            "skipped_reason": None}


ENGINES = {
    "tesseract": run_tesseract,
    "surya": run_surya,
    "paddleocr": run_paddleocr,
}


def run_engine_over_language(
    engine_name: str,
    engine_fn,
    language: str,
    *,
    limit: int | None = None,
    variant: str | None = None,
    per_image_timeout_seconds: float = 60.0,
    raw_root: str | None = None,
    pred_root: str | None = None,
) -> None:
    """
    Runs one engine over every image for one language, appends one
    JSONL line per image. Per-image failures are caught and recorded
    rather than crashing the whole run — a single malformed image
    shouldn't lose the rest of the batch.

    Resumable by default (AGENTS.md Long-running scripts): opens the
    output in append mode, skips any (engine, language, id, variant)
    already present, prints per-image progress, and flushes after each
    write so Ctrl-C leaves a consistent partial file.

    Additionally, engine_fn(image_path, language) is wrapped in a hard
    per-image timeout so a single out-of-distribution image can never
    hang the whole batch indefinitely. On timeout we record
    skipped_reason="timeout" (same jsonl schema as other skips).

    limit and variant are for smoke runs (e.g. first 10 hindi plain
    images) without editing the image directory. raw_root / pred_root
    default to the module roots so a Colab `--data-root` only has to
    be resolved once in main().
    """
    pred = pred_root if pred_root is not None else PRED_ROOT
    out_dir = os.path.join(pred, engine_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{language}.jsonl")

    done = load_completed_keys(out_path)

    images = list(iter_images(language, variant=variant, raw_root=raw_root))
    if limit is not None:
        images = images[:limit]

    todo = [
        (image_id, image_variant, image_path)
        for image_id, image_variant, image_path in images
        if (engine_name, language, image_id, image_variant) not in done
    ]
    n_skip = len(images) - len(todo)
    print(
        f"[{engine_name}/{language}] {len(todo)} to process "
        f"({n_skip} already done, {len(images)} in scope)",
        flush=True,
    )
    if not todo:
        print(f"[{engine_name}/{language}] nothing left -> {out_path}", flush=True)
        return

    with open(out_path, "a", encoding="utf-8") as out_file:
        for i, (image_id, image_variant, image_path) in enumerate(todo, start=1):
            t0 = time.perf_counter()
            print(
                f"[{engine_name}/{language}] {i}/{len(todo)} "
                f"id={image_id} variant={image_variant}",
                flush=True,
            )
            try:
                result = run_engine_with_timeout(
                    engine_fn,
                    image_path,
                    language,
                    timeout_seconds=per_image_timeout_seconds,
                )
            except Exception as e:  # noqa: BLE001 — deliberately broad,
                # see docstring: one bad image must not kill the batch
                result = {"predicted_text": None, "confidence": None,
                           "skipped_reason": f"error: {e}"}

            out_file.write(json.dumps({
                "id": image_id,
                "variant": image_variant,
                "image_path": canonical_image_path(language, image_path),
                "engine": engine_name,
                "language": language,
                **result,
            }, ensure_ascii=False) + "\n")
            out_file.flush()
            # Avoid losing the last line if the process is killed before
            # Python's exit flush; also make progress visible on disk.
            os.fsync(out_file.fileno())
            elapsed = time.perf_counter() - t0
            print(
                f"[{engine_name}/{language}] {i}/{len(todo)} done "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    print(f"[{engine_name}/{language}] done -> {out_path}", flush=True)


def export_predictions_zip(pred_root: str, zip_path: str) -> list[str]:
    """
    Zip every predictions jsonl under pred_root so unzipping at this
    repo's root writes IMPLEMENTATION.md Stage 0's output path:
    data/predictions/{engine}/{language}.jsonl.

    Why a zip, not Drive-only: Colab's files.download() takes a local
    file; a Drive mount is the other allowed export (`--data-root` on
    the mount, skip this). Archive members are the repo paths, not the
    Colab paths, so the person downloading does not have to remap.

    Called from: main(), as the last step when `--export-zip` is set
    (including `--export-only` on a Colab last cell).
    """
    written: list[str] = []
    zip_dir = os.path.dirname(os.path.abspath(zip_path))
    if zip_dir:
        os.makedirs(zip_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, filenames in os.walk(pred_root):
            for name in sorted(filenames):
                if not name.endswith(".jsonl"):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, pred_root).replace(os.sep, "/")
                arcname = f"data/predictions/{rel}"
                zf.write(full, arcname=arcname)
                written.append(arcname)

    print(f"[export] wrote {len(written)} files -> {zip_path}", flush=True)
    for arcname in written:
        print(f"[export]   {arcname}", flush=True)
    print(
        "[export] unzip at the repo root to land files in "
        "data/predictions/ (IMPLEMENTATION.md Stage 0)",
        flush=True,
    )
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR baselines over GlotOCR images")
    parser.add_argument(
        "--language",
        action="append",
        choices=LANGUAGES,
        help="Language(s) to run. Default: all four.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        choices=list(ENGINES.keys()),
        help="Engine(s) to run. Default: all three.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many images per language (sorted order).",
    )
    parser.add_argument(
        "--variant",
        choices=["plain", "degraded"],
        default=None,
        help="If set, only process this variant (plain or degraded).",
    )
    parser.add_argument(
        "--per-image-timeout-seconds",
        type=float,
        default=60.0,
        help="Hard timeout per engine call on a single image. "
        "If exceeded, record skipped_reason='timeout' and continue.",
    )
    parser.add_argument(
        "--data-root",
        dest="data_root",
        default=None,
        help=(
            "Single data root (default: $OCR_DATA_ROOT or data/). "
            "Inputs at {root}/raw, outputs at {root}/predictions."
        ),
    )
    parser.add_argument(
        "--pred-root",
        dest="pred_root",
        default=None,
        help=(
            "Override predictions directory (default: $OCR_PRED_ROOT or "
            "{data-root}/predictions). Isolated smoke tests only."
        ),
    )
    parser.add_argument(
        "--export-zip",
        dest="export_zip",
        default=None,
        help=(
            "After the run, zip predictions for Colab download. Archive "
            "members are data/predictions/{engine}/{language}.jsonl."
        ),
    )
    parser.add_argument(
        "--export-only",
        dest="export_only",
        action="store_true",
        help="Only write --export-zip; do not run engines.",
    )
    args = parser.parse_args()

    if args.export_only and not args.export_zip:
        parser.error("--export-only requires --export-zip")

    data_root, raw_root, pred_root = resolve_data_paths(
        args.data_root, args.pred_root
    )
    print(
        f"[paths] data_root={data_root} raw={raw_root} "
        f"predictions={pred_root}",
        flush=True,
    )

    if not args.export_only:
        languages = args.language or LANGUAGES
        engines = args.engine or list(ENGINES.keys())

        for engine_name in engines:
            for language in languages:
                run_engine_over_language(
                    engine_name,
                    ENGINES[engine_name],
                    language,
                    limit=args.limit,
                    variant=args.variant,
                    per_image_timeout_seconds=args.per_image_timeout_seconds,
                    raw_root=raw_root,
                    pred_root=pred_root,
                )

    if args.export_zip:
        export_predictions_zip(pred_root, args.export_zip)

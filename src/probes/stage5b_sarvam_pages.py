"""
src/probes/stage5b_sarvam_pages.py

Stage 5b paid Extract fetches for the Hindi Probe 5b pool.

Why this file exists
--------------------
Stage 5a already spent ₹17.50 on 35 pages (10 Hindi plains among them).
Rank-correlation transfer needs Sarvam *text* (for grapheme CER) on the
same image_ids as the instrument's GT-likelihood ranking. The remaining
Hindi plains in that 60-id pool, and optionally the 60 degraded
renders (Decision #15), are new SHA-256 cache keys and therefore new
rupees.

This script is the only Stage 5b entry point allowed to call
`SarvamClient.extract_image`. It prints the exact command and cost,
counts uncached pages by hashing files against `data/cache/sarvam/`,
and **refuses to POST** unless `--i-confirm-spend-inr` matches that
cost to the paisa. A dry-run (default) never instantiates the client.

Do not run this without the user confirming the spend in chat first.
The flag is a second lock, not a substitute for that confirmation.

Outputs
-------
  data/probe_results/sarvam_stage5b_pages.jsonl  — append, skip done paths

Examples (copy the printed cost; do not guess):
  python src/probes/stage5b_sarvam_pages.py --set hindi-plain-remaining
  python src/probes/stage5b_sarvam_pages.py --set both \\
      --i-confirm-spend-inr 55.00
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SRC / "eval") not in sys.path:
    sys.path.insert(0, str(_SRC / "eval"))

from eval.transfer_analysis import (  # noqa: E402
    PRICE_INR_PER_PAGE,
    hindi_probe5b_ids,
    load_jsonl,
)

LANGUAGE_CODE = "hi-IN"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_hit(cache_dir: Path, image_path: Path) -> bool:
    return (cache_dir / f"{_sha256_file(image_path)}.json").exists()


def load_already_done(out_path: Path) -> set[str]:
    """Resume set: image_path strings already in the Stage 5b jsonl."""
    done = set()
    for rec in load_jsonl(out_path):
        ip = rec.get("image_path")
        if ip:
            done.add(ip)
    return done


def stage5a_hindi_plain_ids(repo_root: Path) -> set[str]:
    """Hindi image_ids already purchased in Stage 5a (plain only)."""
    path = repo_root / "data/probe_results/sarvam_transfer_probe.jsonl"
    ids = set()
    for rec in load_jsonl(path):
        if rec.get("script") == "hindi":
            ids.add(str(rec["image_id"]))
    return ids


def load_hindi_gt(repo_root: Path) -> dict[str, dict]:
    rows = {}
    path = repo_root / "data/raw/hindi/ground_truth.jsonl"
    for rec in load_jsonl(path):
        rows[str(rec["id"])] = rec
    return rows


def build_tasks(repo_root: Path, page_set: str) -> list[dict]:
    """
    Build Extract tasks for the requested page set.

    hindi-plain-remaining: Probe 5b Hindi ids not in Stage 5a (50 plains).
    hindi-degraded: all 60 Probe 5b ids' *_degraded.png (new bytes).
    both: concatenation, plains first.
    """
    ids = hindi_probe5b_ids(repo_root)
    already_plain = stage5a_hindi_plain_ids(repo_root)
    gt = load_hindi_gt(repo_root)
    tasks: list[dict] = []

    def add(iid: str, variant: str) -> None:
        rec = gt[iid]
        rel = rec["img_plain_path"] if variant == "plain" else rec["img_degraded_path"]
        img_path = repo_root / rel
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        tasks.append(
            {
                "script": "hindi",
                "condition": "real" if variant == "plain" else "degraded",
                "variant": variant,
                "image_id": iid,
                "image_path": str(img_path),
                "ground_truth": rec["text"],
                "language_code": LANGUAGE_CODE,
                "probe": "stage5b",
            }
        )

    want_plain = page_set in {"hindi-plain-remaining", "both"}
    want_deg = page_set in {"hindi-degraded", "both"}
    if want_plain:
        for iid in ids:
            if iid not in already_plain:
                add(iid, "plain")
    if want_deg:
        for iid in ids:
            add(iid, "degraded")
    return tasks


def estimate_new_pages(
    tasks: list[dict],
    out_path: Path,
    cache_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split tasks into jsonl-done, cache-hit (rewrite jsonl, ₹0), and POST.

    A file already in Stage 5b jsonl is skipped entirely. A SHA-256 cache
    hit is not a new rupee even if jsonl is missing. Only cache misses
    cost ₹0.5.
    """
    done = load_already_done(out_path)
    skip_jsonl, cache_only, need_post = [], [], []
    for t in tasks:
        if t["image_path"] in done:
            skip_jsonl.append(t)
            continue
        if _cache_hit(cache_dir, Path(t["image_path"])):
            cache_only.append(t)
        else:
            need_post.append(t)
    return skip_jsonl, cache_only, need_post


def _print_plan(
    page_set: str,
    tasks: list[dict],
    skip_jsonl: list[dict],
    cache_only: list[dict],
    need_post: list[dict],
) -> float:
    cost = len(need_post) * PRICE_INR_PER_PAGE
    print(f"set              : {page_set}")
    print(f"tasks listed     : {len(tasks)}")
    print(f"already in jsonl : {len(skip_jsonl)}")
    print(f"cache hit, ₹0    : {len(cache_only)}")
    print(f"new API POSTs    : {len(need_post)}")
    print(f"estimated ₹      : {cost:.2f}  (₹{PRICE_INR_PER_PAGE}/page)")
    print()
    print("Exact live command after you confirm this cost in chat:")
    print(
        "  export SARVAM_API_KEY=<your_key>"
    )
    print(
        "  python src/probes/stage5b_sarvam_pages.py "
        f"--set {page_set} --i-confirm-spend-inr {cost:.2f}"
    )
    print()
    print("New POST pages:")
    for t in need_post:
        print(f"  {t['variant']:9s}  {t['image_id']:6s}  {t['image_path']}")
    if not need_post:
        print("  (none)")
    return cost


def _payload_to_record(task: dict, api_result: dict) -> dict:
    sarvam_text = (api_result.get("result") or {}).get("full_text") or ""
    return {
        **task,
        "cached": api_result.get("cached"),
        "job_id": api_result.get("job_id"),
        "sarvam_status": api_result.get("status"),
        "sarvam_text": sarvam_text,
        "confidence": api_result.get("confidence"),
        "usage": api_result.get("usage", {}),
    }


def _read_cache_payload(cache_dir: Path, image_path: Path) -> dict:
    sha = _sha256_file(image_path)
    payload = json.loads((cache_dir / f"{sha}.json").read_text(encoding="utf-8"))
    payload["cached"] = True
    return payload


def run_fetch(
    repo_root: Path,
    page_set: str,
    out_path: Path,
    confirm_inr: float | None,
) -> int:
    """
    Dry-run if confirm_inr is None. Live POST only when confirm matches.
    """
    cache_dir = repo_root / "data" / "cache" / "sarvam"
    tasks = build_tasks(repo_root, page_set)
    skip_jsonl, cache_only, need_post = estimate_new_pages(tasks, out_path, cache_dir)
    cost = _print_plan(page_set, tasks, skip_jsonl, cache_only, need_post)

    if confirm_inr is None:
        print("\nStopped: dry-run (no --i-confirm-spend-inr). No API calls.")
        return 0

    expected = round(cost, 2)
    got = round(float(confirm_inr), 2)
    if got != expected:
        print(
            f"\nREFUSING: --i-confirm-spend-inr {got:.2f} != estimated "
            f"₹{expected:.2f}. Re-run dry-run and pass the printed amount.",
            file=sys.stderr,
        )
        return 2

    if need_post:
        from eval.sarvam_client import SarvamClient

        client = SarvamClient(cache_dir=cache_dir)
        client.language = LANGUAGE_CODE
    else:
        client = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pending_write = cache_only + need_post
    post_paths = {t["image_path"] for t in need_post}
    total = len(pending_write)
    with out_path.open("a", encoding="utf-8") as out_f:
        for idx, task in enumerate(pending_write, start=1):
            path = Path(task["image_path"])
            will_post = task["image_path"] in post_paths and not _cache_hit(
                cache_dir, path
            )
            print(
                f"[{idx:3d}/{total}] "
                f"{'POST' if will_post else 'CACHE'}  "
                f"{task['variant']:9s}  {task['image_id']}",
                flush=True,
            )
            try:
                if _cache_hit(cache_dir, path):
                    api_result = _read_cache_payload(cache_dir, path)
                else:
                    if client is None:
                        raise RuntimeError("uncached page but client was not created")
                    api_result = client.extract_image(path)
                record = _payload_to_record(task, api_result)
            except Exception as exc:
                logger.error("failed %s: %s", task["image_id"], exc)
                record = {**task, "error": str(exc), "confidence": None}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
    print(f"Wrote/appended {total} records → {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5b Sarvam Extract fetches. Dry-run by default. "
            "Live calls require --i-confirm-spend-inr matching the printed ₹."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root (not a machine-local data path). Images hang off data/raw.",
    )
    parser.add_argument(
        "--set",
        dest="page_set",
        choices=("hindi-plain-remaining", "hindi-degraded", "both"),
        default="hindi-plain-remaining",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/probe_results/sarvam_stage5b_pages.jsonl"),
    )
    parser.add_argument(
        "--i-confirm-spend-inr",
        type=float,
        default=None,
        help="Must equal dry-run estimated ₹. Absence means dry-run (no POST).",
    )
    args = parser.parse_args(argv)
    return run_fetch(args.repo_root, args.page_set, args.out, args.i_confirm_spend_inr)


if __name__ == "__main__":
    raise SystemExit(main())

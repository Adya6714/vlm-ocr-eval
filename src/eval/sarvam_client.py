"""
src/eval/sarvam_client.py

Thin, cache-once wrapper around the Sarvam Doc-AI Extract endpoint.

Why this file exists and where it fits in the pipeline
-------------------------------------------------------
Stage 5 of IMPLEMENTATION.md asks whether this project's finding—that
confidence is close to prior-dominated and does not track correctness—
"rhymes" with Sarvam's own production system.  To answer that we need
Sarvam's *confidence* numbers, not just accuracy numbers.  Sarvam's
Digitise endpoint returns text but does NOT expose confidence.  Only the
Extract endpoint's `annotations` object exposes a per-field confidence
score.  So we use Extract with a trivially thin schema (one field:
"full_text") to get one confidence number per page, directly comparable
to this project's own `mean_confidence` metric.

This client is called ONLY from sarvam_transfer_probe.py.  Every other
Stage 5 analysis (thresholds, comparisons, correlation tests) runs
offline against data/cache/sarvam/, never against a fresh API call.
That is the only way to stay inside the ~200-page, ~₹100 budget.

API endpoints confirmed from live docs (docs.sarvam.ai, Sept 2026):
  POST  https://api.sarvam.ai/doc-ai/v1/job/extract      → {job_id, status, run_id}
  GET   https://api.sarvam.ai/doc-ai/v1/job/{job_id}/status
        → {job_id, status, pipeline, usage, created_at, updated_at}
  GET   https://api.sarvam.ai/doc-ai/v1/job/{job_id}/results
        → {type:"extract", job_id, status, usage,
           result: {<field>: <value>, …},
           annotations: {<field>: {confidence: float, sources: […]}, …},
           version: int}

Authentication: api-subscription-key HTTP header (never hardcoded here;
read from env var SARVAM_API_KEY or passed explicitly to SarvamClient).

Cache contract (DECISIONS.md #19):
  Cache key = SHA-256 of raw image bytes (content-addressable).
  Cache file = data/cache/sarvam/<sha256>.json
  On cache hit, return cached payload; never call the API.
  On cache miss, submit job, poll until terminal, cache result, return it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants confirmed from docs.sarvam.ai (Sept 2026)
# ---------------------------------------------------------------------------
_BASE_URL = "https://api.sarvam.ai"
_EXTRACT_PATH = "/doc-ai/v1/job/extract"
_STATUS_PATH = "/doc-ai/v1/job/{job_id}/status"
_RESULTS_PATH = "/doc-ai/v1/job/{job_id}/results"

# Terminal statuses per the docs
_TERMINAL = frozenset({"completed", "partially_completed", "failed", "rejected"})

# Single-field schema that yields one confidence number per page.
# Root must be type:object with non-empty properties; every field needs
# type and a non-empty description (docs.sarvam.ai schema rules, Sept 2026).
_EXTRACT_SCHEMA: str = json.dumps({
    "type": "object",
    "properties": {
        "full_text": {
            "type": "string",
            "description": (
                "The complete transcribed text content of this document page, "
                "in reading order, preserving all words and punctuation exactly "
                "as they appear in the original script."
            ),
        }
    },
})

# Poll interval (docs recommend ≥ 5 s for manual polling on Starter plan)
_POLL_INTERVAL_S = 5
# Hard timeout: 10 minutes per job
_JOB_TIMEOUT_S = 600


class SarvamClientError(RuntimeError):
    """Raised when the Sarvam API returns an unexpected response."""


class SarvamClient:
    """
    Cache-once wrapper around the Sarvam Doc-AI Extract endpoint.

    Usage
    -----
    client = SarvamClient()              # reads SARVAM_API_KEY from env
    result = client.extract_image(path)  # returns cached or fresh result

    The `result` dict always contains:
      "cached"      : bool  — True if served from data/cache/sarvam/
      "image_path"  : str
      "sha256"      : str
      "job_id"      : str   — job that produced this result
      "status"      : str   — terminal status ("completed", …)
      "result"      : dict  — {full_text: str}
      "annotations" : dict  — {full_text: {confidence: float, sources: [...]}}
      "usage"       : dict  — {pages_total, pages_processed, …}
      "confidence"  : float — shortcut: annotations["full_text"]["confidence"]

    Raises SarvamClientError on API errors or job failure.
    Does NOT log or print the API key at any point.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str = "data/cache/sarvam",
        poll_interval_s: float = _POLL_INTERVAL_S,
        job_timeout_s: float = _JOB_TIMEOUT_S,
        language: str = "hi-IN",
    ) -> None:
        """
        Parameters
        ----------
        api_key        API key for Sarvam.  If None, reads SARVAM_API_KEY from
                       the environment.  Never logged or printed.
        cache_dir      Local directory for caching raw JSON results.
                       Each file is named {sha256_of_image_bytes}.json.
        poll_interval_s  Seconds between status polls (docs recommend ≥ 5 s).
        job_timeout_s  Hard timeout per job in seconds.
        language       BCP-47 language hint sent to the API (e.g. "hi-IN").
        """
        key = api_key or os.environ.get("SARVAM_API_KEY")
        if not key:
            raise EnvironmentError(
                "SARVAM_API_KEY is not set in the environment.  "
                "Export it before running: export SARVAM_API_KEY=<your_key>"
            )
        # Store as private; never expose via repr or str
        self._api_key = key

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.poll_interval_s = poll_interval_s
        self.job_timeout_s = job_timeout_s
        self.language = language

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_image(self, image_path: Path | str) -> dict[str, Any]:
        """
        Return Extract results for image_path, hitting the cache if possible.

        This is the single entry point that sarvam_transfer_probe.py calls.
        Every re-run for the same image returns the cached payload without
        an API call.  The cache key is SHA-256 of the raw file bytes so
        renaming a file never causes a re-fetch.

        Parameters
        ----------
        image_path  Path to a PNG or JPEG image file on disk.

        Returns
        -------
        dict  (see class docstring for keys)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        raw_bytes = image_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        cache_path = self.cache_dir / f"{sha256}.json"

        if cache_path.exists():
            logger.info("cache hit  %s  (%s)", sha256[:12], image_path.name)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["cached"] = True
            return payload

        logger.info("cache miss %s  (%s) — calling API", sha256[:12], image_path.name)
        payload = self._call_api(image_path, raw_bytes, sha256)
        # Write to cache before returning so a crash mid-analysis
        # never costs a repeat API call.
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("cached     %s  → %s", sha256[:12], cache_path.name)
        payload["cached"] = False
        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Return auth headers.  Key is never logged."""
        return {"api-subscription-key": self._api_key}

    def _call_api(
        self, image_path: Path, raw_bytes: bytes, sha256: str
    ) -> dict[str, Any]:
        """
        Submit an Extract job, poll until terminal, fetch and return results.

        The returned dict is exactly what gets written to cache — it does NOT
        include the "cached" flag (that is added by the caller).
        """
        job_id, initial_status = self._submit_job(image_path, raw_bytes)
        logger.info("job submitted  job_id=%s  status=%s", job_id, initial_status)

        if initial_status not in _TERMINAL:
            terminal_status = self._poll_until_terminal(job_id)
        else:
            terminal_status = initial_status

        if terminal_status in ("failed", "rejected"):
            raise SarvamClientError(
                f"Sarvam job {job_id} ended with status={terminal_status!r} "
                f"for image {image_path.name!r}"
            )

        results = self._fetch_results(job_id)

        # Build a clean, self-contained cache payload
        confidence = self._extract_confidence(results)
        return {
            "image_path": str(image_path),
            "sha256": sha256,
            "job_id": job_id,
            "status": terminal_status,
            "result": results.get("result", {}),
            "annotations": results.get("annotations", {}),
            "usage": results.get("usage", {}),
            "confidence": confidence,
        }

    def _submit_job(self, image_path: Path, raw_bytes: bytes) -> tuple[str, str]:
        """
        POST to the Extract endpoint.

        Returns (job_id, initial_status).

        Request shape confirmed from docs.sarvam.ai Extract reference (Sept 2026):
          multipart/form-data  with fields:
            file     — the image bytes
            schema   — JSON string (NOT a dict; must be json.dumps'd)
            language — BCP-47 code
            output_format — "json"
        """
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        files = {
            "file": (image_path.name, raw_bytes, mime),
        }
        data = {
            "schema": _EXTRACT_SCHEMA,
            "language": self.language,
            "output_format": "json",
        }
        url = _BASE_URL + _EXTRACT_PATH
        resp = requests.post(url, headers=self._headers(), files=files, data=data, timeout=60)
        self._raise_for_status(resp, "extract")
        body = resp.json()
        return body["job_id"], body["status"]

    def _poll_until_terminal(self, job_id: str) -> str:
        """
        Poll GET /doc-ai/v1/job/{job_id}/status at _POLL_INTERVAL_S cadence
        until a terminal status is reached or _JOB_TIMEOUT_S elapses.

        Terminal statuses (docs.sarvam.ai, Sept 2026):
            completed, partially_completed, failed, rejected
        """
        url = _BASE_URL + _STATUS_PATH.format(job_id=job_id)
        deadline = time.monotonic() + self.job_timeout_s
        while True:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            self._raise_for_status(resp, "status")
            body = resp.json()
            status = body.get("status", "")
            logger.debug("poll  job_id=%s  status=%s  usage=%s",
                         job_id, status, body.get("usage"))
            if status in _TERMINAL:
                return status
            if time.monotonic() >= deadline:
                raise SarvamClientError(
                    f"Timed out after {self.job_timeout_s}s waiting for "
                    f"job {job_id} (last status: {status!r})"
                )
            time.sleep(self.poll_interval_s)

    def _fetch_results(self, job_id: str) -> dict[str, Any]:
        """
        GET /doc-ai/v1/job/{job_id}/results

        Response shape confirmed from docs.sarvam.ai results reference (Sept 2026):
          {type:"extract", job_id, status, usage,
           result: {full_text: str},
           annotations: {full_text: {confidence: float, sources:[…]}},
           version: int}
        """
        url = _BASE_URL + _RESULTS_PATH.format(job_id=job_id)
        resp = requests.get(url, headers=self._headers(), timeout=60)
        self._raise_for_status(resp, "results")
        return resp.json()

    @staticmethod
    def _extract_confidence(results: dict[str, Any]) -> float:
        """
        Pull confidence out of the annotations object.

        The annotations object mirrors the result shape; every leaf has:
          {confidence: float (0-1), sources: [{document_id, filename, page_num}]}

        We use full_text.confidence as our single "Sarvam confidence" for a page,
        directly comparable to the instrument's mean_confidence metric.
        Falls back to 0.0 if the field is absent (e.g. extraction found nothing).
        """
        try:
            return float(results["annotations"]["full_text"]["confidence"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Could not extract full_text confidence; defaulting to 0.0")
            return 0.0

    @staticmethod
    def _raise_for_status(resp: requests.Response, ctx: str) -> None:
        """Raise SarvamClientError with a useful message on HTTP errors."""
        if not resp.ok:
            raise SarvamClientError(
                f"Sarvam API [{ctx}] returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

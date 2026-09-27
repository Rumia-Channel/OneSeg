"""Persist structured RF/DSP diagnostics without blocking RTL-SDR USB reads.

One JSON object per line: UTC time, elapsed seconds, schema, session and
window identifiers, event kind, observed measurements and individual checks.
A small bounded queue only carries metadata; no IQ samples or TS packets
are ever written to these logs. A separate writer flushes after every line.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import perf_counter
from uuid import uuid4

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def default_log_directory() -> Path:
    return Path.home() / "OneSeg" / "logs"


def check_window(report: dict) -> dict[str, dict]:
    """Evidence-based stage checks; 'unknown' is never upgraded to PASS.

    A 0x47 sync byte or pilot correlation alone cannot prove ISDB-T TS.
    A failed TMCC check makes later stages 'unknown' instead of a false
    assertion that Viterbi/RS itself failed.
    """
    checks = {}

    def check(key, value, good, bad, good_msg, bad_msg):
        if value is None:
            checks[key] = {"status": "unknown", "reason": "not measured"}
        elif good(value):
            checks[key] = {"status": "pass", "reason": good_msg}
        elif bad(value):
            checks[key] = {"status": "fail", "reason": bad_msg}
        else:
            checks[key] = {"status": "warn", "reason": bad_msg}

    clip = report.get("fullscale_percent")
    check(
        "adc_clipping", clip,
        lambda x: x < 1.0, lambda x: x >= 5.0,
        "less than 1% exact full-scale I/Q occupancy",
        "1-5% marginal or >=5% overloaded; exact occupancy is not BER",
    )
    check(
        "cp", report.get("cp_quality"),
        lambda x: x >= .65, lambda x: x < .4,
        "repeated OFDM CP correlation candidate",
        "weak/uncertain OFDM CP evidence; CP alone is not TS lock",
    )
    check(
        "pilots", report.get("pilot_coherence"),
        lambda x: x >= .75, lambda x: x < .45,
        "central pilot alignment candidate",
        "low/uncertain central pilot coherence; not TMCC lock",
    )
    count = report.get("tmcc_parity_verified_frames")
    check(
        "tmcc_parity", count, lambda x: x >= 1, lambda x: x == 0,
        "at least one protected zero-syndrome TMCC frame",
        "no 82-bit parity-verified TMCC frame",
    )
    accepted = report.get("accepted_chunk") if count is not None else None
    check(
        "rs_ts", accepted, lambda x: x >= 1, lambda x: x == 0,
        "one or more genuine RS-verified 188-byte packets",
        "no genuine TS packet recovered in this window",
    )
    check(
        "pat", report.get("has_pat"),
        lambda x: x is True, lambda x: x is False,
        "CRC-validated PAT found", "PAT absent from verified partial TS",
    )
    check(
        "pmt", report.get("has_pmt"),
        lambda x: x is True, lambda x: x is False,
        "CRC-validated PMT found", "PMT absent from verified partial TS",
    )
    duration = (
        report.get("decoder_window_seconds")
        if not report.get("failure_reason") else None
    )
    seconds = report.get("window_seconds", 3.0)
    check(
        "dsp_sustainable", duration,
        lambda x: x <= seconds, lambda x: x > seconds,
        "decoder wall time no greater than acquired RF duration",
        "decoder takes longer than RF acquisition; backlog will grow",
    )
    return checks


class SessionTelemetry:
    """One append-only JSONL session; event submission never waits for disk."""

    def __init__(self, directory: Path | None = None, *, capacity: int = 2048):
        if capacity < 1:
            raise ValueError("telemetry capacity must be positive")
        directory = Path(directory) if directory is not None else default_log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            + "_" + uuid4().hex[:8]
        )
        self.path = directory / f"oneseg_live_{self.session_id}.jsonl"
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self._queue: Queue[dict | None] = Queue(maxsize=capacity)
        self._start = perf_counter()
        self._lock = Lock()
        self._dropped = 0
        self._closed = False
        self.error: str | None = None
        self._writer = Thread(
            target=self._run, name="OneSeg-Telemetry-Writer", daemon=False,
        )
        self._writer.start()

    def emit(self, event: str, *, window: int | None = None, **measurements) -> bool:
        with self._lock:
            if self._closed:
                return False
            record = {
                "schema": SCHEMA_VERSION,
                "session": self.session_id,
                "utc": utc_now(),
                "elapsed_s": round(perf_counter() - self._start, 3),
                "event": event,
                "window": window,
                **measurements,
            }
            try:
                self._queue.put_nowait(record)
            except Full:
                self._dropped += 1
                return False
            return True

    def _run(self):
        try:
            while True:
                entry = self._queue.get()
                if entry is None:
                    break
                self._file.write(
                    json.dumps(entry, ensure_ascii=False, allow_nan=False, default=str)
                    + "\n"
                )
                self._file.flush()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._dropped:
                # Can block only on final stop, never in USB acquisition.
                self._queue.put({
                    "schema": SCHEMA_VERSION, "session": self.session_id,
                    "utc": utc_now(), "elapsed_s": round(
                        perf_counter() - self._start, 3
                    ),
                    "event": "telemetry_queue_overflow",
                    "window": None, "dropped_events": self._dropped,
                })
            self._queue.put(None)
        self._writer.join()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def summarize_log(path: Path) -> dict:
    """Aggregate one session, including failures and lack of measurements."""
    path = Path(path)
    counts = Counter()
    by_gain = defaultdict(
        lambda: {
            "windows": 0, "accepted_ts": 0, "failed": 0,
            "fullscale_percent": [], "cp": [], "pilots": [],
            "dsp_seconds": [], "verified_tmcc_frames": 0,
        }
    )
    checks = defaultdict(Counter)
    observed = 0
    final = None
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # A crash can leave only the last line incomplete. Earlier
                # independent JSONL entries remain valuable evidence.
                if number and not stream.read(1):
                    counts["truncated_last_line"] += 1
                    break
                raise ValueError(f"invalid JSONL line {number}") from exc
            if record.get("schema") != SCHEMA_VERSION:
                raise ValueError(f"unsupported log schema at line {number}")
            counts[record["event"]] += 1
            final = record["utc"]
            if record["event"] != "window_result":
                continue
            observed += 1
            data = record.get("metrics", {})
            gain = data.get("applied_gain_db")
            key = "unknown" if gain is None else f"{gain:g}"
            slot = by_gain[key]
            slot["windows"] += 1
            slot["accepted_ts"] += int(data.get("accepted_chunk") or 0)
            slot["failed"] += int(bool(data.get("failure_reason")))
            slot["verified_tmcc_frames"] += int(
                data.get("tmcc_parity_verified_frames") or 0
            )
            for key2, val in (
                ("fullscale_percent", data.get("fullscale_percent")),
                ("cp", data.get("cp_quality")),
                ("pilots", data.get("pilot_coherence")),
                ("dsp_seconds", data.get("decoder_window_seconds")),
            ):
                if val is not None:
                    slot[key2].append(val)
            for name, item in record.get("checks", {}).items():
                checks[name][item["status"]] += 1
    return {
        "path": str(path),
        "events": dict(counts),
        "window_results": observed,
        "latest_utc": final,
        "by_applied_gain": dict(by_gain),
        "checks": {key: dict(value) for key, value in checks.items()},
        "not_a_continuous_tv_lock": True,
    }

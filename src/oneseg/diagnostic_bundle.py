"""Automatically retain bounded reproducible live RF evidence on the user's PC.

No network upload. At most ONE first failed and ONE first successful
3-second interleaved-u8 IQ window per session, plus authentic partial TS.
Other windows have metadata only in the existing JSONL. Packaging is
done after USB acquisition and decoder shutdown, never in the USB loop.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from zipfile import ZIP_STORED, ZipFile

from .dsp import DEFAULT_SAMPLE_RATE
from .telemetry import summarize_log

MAX_CAPTURE_BYTES = 2 * DEFAULT_SAMPLE_RATE * 3  # 12,288,000 bytes
EXAMPLE_TYPES = ("first_failure", "first_success")


class DiagnosticBundle:
    """Bounded, on-disk examples selected by RESULT, not raw RF power."""

    def __init__(self, log: Path, *, max_capture_bytes: int = MAX_CAPTURE_BYTES):
        self.log = Path(log)
        if max_capture_bytes < 2:
            raise ValueError("invalid capture size limit")
        self.max_capture_bytes = max_capture_bytes
        self.directory = self.log.with_suffix(".diagnostics")
        self.directory.mkdir(mode=0o700, exist_ok=False)
        self.bundle_path = self.log.with_suffix(".diagnostics.zip")
        if self.bundle_path.exists():
            raise FileExistsError(self.bundle_path)
        self.examples: dict[str, dict] = {}
        self.copy_errors: list[str] = []

    def save_example(
        self, category: str, raw: Path, *, window: int, metrics: dict,
        ts: Path | None = None,
    ) -> dict | None:
        """Copy a representative raw signal after decoder finishes this window.

        Never let a disk-full or missing TS file erase the main RF result.
        A .u8iq is *unsigned interleaved I,Q*, not complex64 or MPEG-TS.
        """
        if category not in EXAMPLE_TYPES:
            raise ValueError("unknown diagnostic category")
        if category in self.examples:
            return None
        raw = Path(raw)
        destination = self.directory / f"{category}_window_{window:06d}.u8iq"
        sidecar = destination.with_suffix(".u8iq.json")
        packet_file = destination.with_suffix(".ts")
        try:
            size = raw.stat().st_size
            if size == 0 or size % 2 or size > self.max_capture_bytes:
                raise ValueError("invalid or oversized raw I/Q example")
            expected = round(
                float(metrics.get("window_seconds", 3.0))
                * DEFAULT_SAMPLE_RATE * 2
            )
            if size != expected:
                raise ValueError(
                    f"raw I/Q size {size} does not match window {expected}"
                )
            shutil.copyfile(raw, destination)
            if destination.stat().st_size != size:
                raise IOError("incomplete saved raw I/Q example")
            source_ts = Path(ts) if ts is not None else None
            if (
                category == "first_success"
                and source_ts is not None and source_ts.is_file()
            ):
                count = source_ts.stat().st_size
                if count and count % 188 == 0:
                    shutil.copyfile(source_ts, packet_file)
            detail = {
                "format": "unsigned_interleaved_iq_8bit",
                "sample_rate_hz": DEFAULT_SAMPLE_RATE,
                "sample_count": size // 2,
                "byte_order": "I0,Q0,I1,Q1,...",
                "center_frequency_hz": metrics.get("center_frequency_hz"),
                "requested_gain_db": metrics.get("requested_gain_db"),
                "applied_gain_db": metrics.get("applied_gain_db"),
                "ppm": metrics.get("ppm"),
                "capture_window": window,
                "example_category": category,
                "window_metrics": metrics,
                "sample_continuity_verified": False,
                "mpeg_ts_recovered_from_this_window": category == "first_success",
                "no_auto_upload": True,
            }
            sidecar.write_text(
                json.dumps(detail, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            record = {
                "category": category,
                "window": window,
                "iq": destination.name,
                "metadata": sidecar.name,
                "ts": packet_file.name if packet_file.exists() else None,
                "iq_bytes": size,
            }
            self.examples[category] = record
            return record
        except Exception as exc:
            self.copy_errors.append(
                f"{category} window {window}: {type(exc).__name__}: {exc}"
            )
            for file in (destination, sidecar, packet_file):
                file.unlink(missing_ok=True)
            return None

    def finish(self, *, survey_report: Path | None = None) -> Path:
        """Package metadata and bounded I/Q, then discard loose copies.

        JSONL is already flushed and closed by caller. If packaging fails,
        leave loose files and JSONL intact; no observed evidence is deleted.
        """
        summary = {
            "format": "oneseg_live_diagnostics_v1",
            "log": self.log.name,
            "session_summary": summarize_log(self.log),
            "saved_examples": list(self.examples.values()),
            "save_errors": self.copy_errors,
            "no_auto_upload": True,
            "raw_iq_is_not_complex64": True,
            "real_time_tv_verified": False,
        }
        report = self.directory / "report.json"
        report.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary = self.bundle_path.with_name(
            self.bundle_path.name + ".partial"
        )
        try:
            with ZipFile(temporary, mode="x", compression=ZIP_STORED) as archive:
                archive.write(self.log, arcname="session.jsonl")
                if survey_report is not None:
                    archive.write(Path(survey_report), arcname="survey.json")
                for file in sorted(self.directory.iterdir()):
                    if file.is_file():
                        archive.write(file, arcname=file.name)
            temporary.replace(self.bundle_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        shutil.rmtree(self.directory)
        return self.bundle_path

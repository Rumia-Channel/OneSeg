import json
from pathlib import Path
from zipfile import ZipFile

from oneseg.diagnostic_bundle import DiagnosticBundle
from oneseg.telemetry import SessionTelemetry


def _metrics(seconds=1.0):
    return {
        "window_seconds": seconds,
        "center_frequency_hz": 515142857, "ppm": 0,
        "requested_gain_db": -5.0, "applied_gain_db": -5.4,
        "cp_quality": .91, "pilot_coherence": .75,
        "fullscale_percent": .2, "accepted_chunk": 0,
    }


def test_auto_bundle_has_raw_reproducible_fail_and_success_plus_jsonl(tmp_path):
    with SessionTelemetry(tmp_path) as log:
        bundle = DiagnosticBundle(log.path)
        data = tmp_path / "capture.u8iq"
        data.write_bytes(bytes((111, 127)) * 2_048_000)
        failure = bundle.save_example(
            "first_failure", data, window=3, metrics=_metrics()
        )
        assert failure["iq_bytes"] == 4_096_000
        assert bundle.save_example(
            "first_failure", data, window=4, metrics=_metrics()
        ) is None
        ts = tmp_path / "sample.ts"
        ts.write_bytes((b"\x47" + bytes(187)) * 4)
        success = bundle.save_example(
            "first_success", data, window=5,
            metrics={**_metrics(), "accepted_chunk": 4}, ts=ts
        )
        assert success["ts"] is not None
        log.emit("window_result", window=3, metrics=_metrics(), checks={})
        log.emit("window_result", window=5,
                 metrics={**_metrics(), "accepted_chunk": 4}, checks={})
    output = bundle.finish()
    assert output.exists()
    assert not bundle.directory.exists()
    with ZipFile(output) as archive:
        filenames = set(archive.namelist())
        assert "session.jsonl" in filenames
        assert "report.json" in filenames
        assert sum(n.endswith(".u8iq") for n in filenames) == 2
        assert sum(n.endswith(".ts") for n in filenames) == 1
        raw = archive.read(failure["iq"])
        assert raw[:6] == bytes((111,127))*3
        assert len(raw) == 4_096_000
        metadata = json.loads(archive.read(failure["metadata"]))
        assert metadata["format"] == "unsigned_interleaved_iq_8bit"
        assert metadata["sample_count"] == 2_048_000
        assert metadata["sample_continuity_verified"] is False
        report = json.loads(archive.read("report.json"))
        assert len(report["saved_examples"]) == 2
        assert report["session_summary"]["window_results"] == 2


def test_invalid_raw_record_does_not_destroy_jsonl_or_write_bad_example(tmp_path):
    with SessionTelemetry(tmp_path) as log:
        bundle = DiagnosticBundle(log.path)
        raw = tmp_path / "bad.u8iq"
        raw.write_bytes(bytes(3))
        assert bundle.save_example(
            "first_failure", raw, window=0, metrics=_metrics()
        ) is None
        assert len(bundle.copy_errors) == 1
        log.emit("session_error", error="fake USB exception")
    with ZipFile(bundle.finish()) as archive:
        assert json.loads(archive.read("report.json"))["save_errors"]
        assert "session_error" in archive.read("session.jsonl").decode()
        assert not any(n.endswith(".u8iq") for n in archive.namelist())

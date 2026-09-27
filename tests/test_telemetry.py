import json
from threading import Thread

import pytest
from oneseg.telemetry import SessionTelemetry, check_window, summarize_log


def test_jsonl_persists_failure_and_recovery_with_checks(tmp_path):
    with SessionTelemetry(tmp_path) as logger:
        first = {
            "fullscale_percent": 8.88, "cp_quality": .95,
            "pilot_coherence": .947, "tmcc_parity_verified_frames": 0,
            "accepted_chunk": 0, "has_pat": False, "has_pmt": False,
            "decoder_window_seconds": .96, "window_seconds": 3.0,
            "applied_gain_db": -5.4,
            "failure_reason": "no parity-verified TMCC",
        }
        logger.emit("session_start", frequency_hz=515142857, gain_db=-5.4)
        logger.emit(
            "window_result", window=0, metrics=first,
            checks=check_window(first),
        )
        second = {
            **first, "fullscale_percent": 1.96,
            "tmcc_parity_verified_frames": 2,
            "accepted_chunk": 112, "failure_reason": None,
            "applied_gain_db": -7.3,
        }
        logger.emit(
            "window_result", window=1, metrics=second,
            checks=check_window(second),
        )
        logger.emit("session_stop", accepted_total=112)
    events = [json.loads(x) for x in logger.path.read_text().splitlines()]
    assert [entry["event"] for entry in events] == [
        "session_start", "window_result", "window_result", "session_stop"
    ]
    assert all(entry["utc"].endswith("+00:00") for entry in events)
    assert events[1]["checks"]["adc_clipping"]["status"] == "fail"
    assert events[1]["checks"]["tmcc_parity"]["status"] == "fail"
    assert events[2]["checks"]["tmcc_parity"]["status"] == "pass"
    summary = summarize_log(logger.path)
    assert summary["window_results"] == 2
    assert summary["by_applied_gain"]["-5.4"]["failed"] == 1
    assert summary["by_applied_gain"]["-7.3"]["accepted_ts"] == 112
    assert not logger.emit("after_stop")


def test_auto_checks_do_not_treat_missing_as_success():
    statuses = {k: v["status"] for k,v in check_window({}).items()}
    assert set(statuses.values()) == {"unknown"}
    results = check_window({
        "cp_quality": .95, "pilot_coherence": .95,
        "tmcc_parity_verified_frames": 0,
        "fullscale_percent": .12, "decoder_window_seconds": .96,
    })
    assert results["cp"]["status"] == "pass"
    assert results["pilots"]["status"] == "pass"
    assert results["tmcc_parity"]["status"] == "fail"
    assert results["rs_ts"]["status"] == "unknown"


def test_concurrent_event_submission_is_valid_json_lines(tmp_path):
    with SessionTelemetry(tmp_path, capacity=300) as logger:
        threads = [
            Thread(target=lambda i=i: [
                logger.emit("status", window=i, sample=n)
                for n in range(25)
            ])
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert not logger.error
    events = [json.loads(x) for x in logger.path.read_text().splitlines()]
    assert len(events) == 100
    assert all(e["event"] == "status" for e in events)


def test_reject_bad_capacity_and_invalid_log(tmp_path):
    with pytest.raises(ValueError):
        SessionTelemetry(tmp_path, capacity=0)
    invalid = tmp_path/"bad.jsonl"
    invalid.write_text("{bad}\n{}\n")
    with pytest.raises(ValueError):
        summarize_log(invalid)



def test_old_survey_log_with_false_zero_getter_is_regrouped(tmp_path):
    """Old 2026-09-27 JSONL must not silently collapse all gains into zero."""
    with SessionTelemetry(tmp_path) as logger:
        logger.emit(
            "survey_gain_plan",
            supported_gains_db=[-9.9, -7.3, -6.5, -5.4],
        )
        for index, (gain, packets) in enumerate(
            ((-9.9, 261), (-7.3, 142), (-6.5, 106), (-6.5, 269))
        ):
            logger.emit(
                "window_result", window=index,
                metrics={
                    "gain_requested_db": gain,
                    "applied_gain_db": 0.0,
                    "accepted_chunk": packets,
                    "tmcc_parity_verified_frames": 2,
                    "fullscale_percent": 1.1,
                },
                checks={},
            )
    summary = summarize_log(logger.path)
    assert summary["legacy_gain_readback_mismatches_regrouped"] == 4
    assert summary["by_applied_gain"]["-9.9"]["accepted_ts"] == 261
    assert summary["by_applied_gain"]["-7.3"]["accepted_ts"] == 142
    assert summary["by_applied_gain"]["-6.5"]["accepted_ts"] == 375
    assert "0" not in summary["by_applied_gain"]

import json
from pathlib import Path

import pytest
from oneseg.decode import decode_capture


def test_end_to_end_orchestration_without_usb_or_fake_ts(tmp_path, monkeypatch):
    cap = tmp_path / "sample.c64"
    cap.write_bytes(b"\x00" * 8)
    cap.with_suffix(".c64.json").write_text('{"sample_rate_hz": 2048000}')
    stages = []

    def fake_analyze(source, *, seconds, layer_a_output, compress_fixture):
        assert source == cap and seconds == 3
        assert compress_fixture is False
        assert not layer_a_output.exists()
        layer_a_output.write_bytes(b"layer")
        stages.append("pilot")
        return {
            "bch_parity_verified_frames": [{
                "layer_A": {"modulation":"QPSK", "code_rate":"2/3",
                            "segments":1, "time_interleaving_mode3":4}
            }],
            "mode_assumed":3, "guard_assumed":"1/8",
            "iq_fullscale_percent": 11.3,
            "iq_rms": 0.5,
            "cp_quality": 0.98,
            "pilot_coherence": 0.95,
            "iq_overload_warning": True,
        }

    def fake_deint(layer, report, output, *, compress):
        assert compress is False
        assert layer.read_bytes() == b"layer"
        assert json.loads(report.read_text())["mode_assumed"] == 3
        output.write_bytes(b"bits")
        stages.append("deinterleave")
        return {"output_bit_count": 1234}

    def fake_recover(fixture, output, *, max_ofdm_symbols):
        assert fixture.read_bytes() == b"bits"
        assert max_ofdm_symbols == 1020
        output.write_bytes(b"\x47" + b"\xff" * 187)
        output.with_suffix(".ts.json").write_text("{}")
        stages.append("recover")
        return {
            "rs_and_ts_accepted_packets": 1,
            "rejected_rs_or_invalid_ts_packets": 0,
            "pat_programs": {},
        }

    monkeypatch.setattr("oneseg.decode.analyze_capture", fake_analyze)
    monkeypatch.setattr("oneseg.decode.process_fixture", fake_deint)
    monkeypatch.setattr("oneseg.decode.recover_file", fake_recover)
    messages = []
    output = tmp_path / "result.ts"
    data = decode_capture(cap, output, progress=messages.append)
    assert stages == ["pilot", "deinterleave", "recover"]
    assert output.stat().st_size == 188
    assert data["tmcc_parity_verified_frames"] == 1
    assert data["input_overload_warning"]
    assert data["may_not_be_playable_without_pat_pmt"]
    assert "temporary stage" in json.loads(
        output.with_suffix(".ts.json").read_text()
    )["source_npz"]
    assert len(messages) == 4
    with pytest.raises(FileExistsError):
        decode_capture(cap, output)


def test_tmcc_failure_carries_quantified_signal_evidence(tmp_path, monkeypatch):
    from oneseg.decode import UnverifiedTmccError

    source = tmp_path / "weak.c64"
    source.write_bytes(bytes(8))
    source.with_suffix(".c64.json").write_text("{}")

    def no_tmcc(source, *, seconds, layer_a_output, compress_fixture):
        return {
            "bch_parity_verified_frames": [],
            "cp_quality": 0.91,
            "pilot_coherence": 0.34,
            "iq_rms": 0.014,
            "iq_fullscale_percent": 0.12,
            "single_sync_candidates": 2,
            "repeated_sync_candidates": [],
            "gain_db": -9.9,
        }

    monkeypatch.setattr("oneseg.decode.analyze_capture", no_tmcc)
    with pytest.raises(UnverifiedTmccError) as problem:
        decode_capture(source, tmp_path / "never.ts")
    report = problem.value.diagnostics
    assert report["cp_quality"] == pytest.approx(0.91)
    assert report["pilot_coherence"] == pytest.approx(0.34)
    assert "I/Q RMS 0.0140" in str(problem.value)
    assert "full-scale 0.12%" in str(problem.value)
    assert not (tmp_path / "never.ts").exists()


def test_rejects_missing_sidecar_and_bad_extension(tmp_path):
    source = tmp_path / "iq.c64"
    source.write_bytes(b"\0" * 8)
    with pytest.raises(FileNotFoundError):
        decode_capture(source, tmp_path / "result.ts")
    source.with_suffix(".c64.json").write_text("{}")
    with pytest.raises(ValueError):
        decode_capture(source, tmp_path / "result.txt")
    with pytest.raises(ValueError):
        decode_capture(source, tmp_path / "result.ts", seconds=10)


@pytest.mark.parametrize("value", [None, "unknown", float("nan"), float("inf"), True])
def test_incomplete_tmcc_diagnostics_do_not_mask_failure(value):
    from oneseg.decode import DecodeStageError, UnverifiedTmccError

    report = {"cp_quality": value, "bch_parity_verified_frames": None,
              "repeated_sync_candidates": None}
    error = UnverifiedTmccError(report)
    assert "CP unavailable" in str(error)
    assert error.diagnostics["cp_quality"] is value
    stage_error = DecodeStageError("deinterleave", report, ValueError("bad input"))
    assert str(stage_error) == "deinterleave failed: bad input"
    assert stage_error.diagnostics["original_error"] == "bad input"


def test_stage_failure_retains_original_cause_and_diagnostics(tmp_path, monkeypatch):
    from oneseg.decode import DecodeStageError

    source = tmp_path / "source.c64"
    source.write_bytes(bytes(8))
    source.with_suffix(".c64.json").write_text("{}")
    monkeypatch.setattr("oneseg.decode.analyze_capture", lambda *args, **kwargs: {
        "bch_parity_verified_frames": [{"layer_A": {}}], "cp_quality": 0.8,
    })
    def fail(*args, **kwargs):
        raise ValueError("unmatched protected metadata")
    monkeypatch.setattr("oneseg.decode.process_fixture", fail)
    output = tmp_path / "never.ts"
    with pytest.raises(DecodeStageError) as failure:
        decode_capture(source, output)
    assert failure.value.stage == "deinterleave"
    assert failure.value.diagnostics["tmcc_parity_verified_frames"] == 1
    assert str(failure.value.__cause__) == "unmatched protected metadata"
    assert not output.exists()

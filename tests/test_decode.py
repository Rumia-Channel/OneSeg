import json
from pathlib import Path

import pytest
from oneseg.decode import decode_capture


def test_end_to_end_orchestration_without_usb_or_fake_ts(tmp_path, monkeypatch):
    cap = tmp_path / "sample.c64"
    cap.write_bytes(b"\x00" * 8)
    cap.with_suffix(".c64.json").write_text('{"sample_rate_hz": 2048000}')
    stages = []

    def fake_analyze(source, *, seconds, layer_a_output):
        assert source == cap and seconds == 3
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
            "iq_overload_warning": True,
        }

    def fake_deint(layer, report, output):
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

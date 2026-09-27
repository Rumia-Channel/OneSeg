"""RF extraction saves real stages even on TMCC/Viterbi failures."""
import json

import pytest
from oneseg.extract import extract_channel


class FakeTuner:
    valid_gains_db = [-9.9, -7.3, -6.5, -5.4, 5.8]

    def __init__(self):
        self.sample_rate = None
        self.center_freq = None
        self.freq_correction = 0
        self.commanded_gain = None
        self.closed = False
        self.raw_calls = 0

    @property
    def gain(self):
        return 0.0  # Real negative gain readback failure on FC0013

    @gain.setter
    def gain(self, value):
        self.commanded_gain = value

    def close(self):
        self.closed = True

    def read_samples_async(self, *a, **k):
        raise AssertionError("unsafe native async must not be used")


def _install_fake_stages(monkeypatch, tuner, *, verified=True, fail_fec=False):
    call_order = []

    def fake_record(device, path, *, samples_required, cancelled, warmup_buffers):
        assert device is tuner and not device.closed
        assert device.commanded_gain == -6.5
        assert samples_required == 2048000
        call_order.append("capture")
        device.raw_calls += 1
        if device.raw_calls == 2 and fail_fec == "usb":
            raise IOError("short raw USB buffer")
        path.write_bytes(bytes((128, 127)) * 64)
        return 64

    def fake_expand(raw, target, *, metadata):
        assert tuner.closed
        assert metadata["gain_db"] == -6.5
        target.write_bytes(bytes(8) * 64)
        target.with_suffix(".c64.json").write_text("{}")
        call_order.append("expand")

    def fake_quality(c64):
        call_order.append("quality")
        return {"samples": 64, "full_scale_fraction": .1, "rms": .3}

    def fake_analyze(c64, *, seconds, layer_a_output, compress_fixture):
        assert tuner.closed and not compress_fixture
        layer_a_output.write_bytes(b"real intermediate mock")
        call_order.append("pilot")
        return {
            "cp_quality": .98,
            "pilot_coherence": .97,
            "bch_parity_verified_frames": (
                [{"syndrome": 0}] if verified else []
            ),
            "tmcc_soft_source": "combined",
            "tmcc_per_carrier_verified_counts": {"combined":int(verified)},
            "iq_rms": .32, "iq_fullscale_percent": .2,
        }

    def fake_deinterleave(layer, tmcc, target, *, compress):
        assert tuner.closed and layer.is_file() and tmcc.is_file()
        assert not compress
        target.write_bytes(b"bit intermediate mock")
        call_order.append("deinterleave")
        return {"output_bit_count": 7000}

    def fake_recover(bit_fixture, target, *, max_ofdm_symbols):
        assert max_ofdm_symbols == 2200 and bit_fixture.is_file()
        if fail_fec is True:
            raise ValueError("no reliable 204-byte 0x47 alignment")
        target.write_bytes((b"\x47" + bytes(187)) * 11)
        target.with_suffix(".ts.json").write_text("{}")
        call_order.append("recover")
        return {
            "rs_and_ts_accepted_packets": 11,
            "rejected_rs_or_invalid_ts_packets": 3,
            "pat_programs": {},
            "pmt_elementary_streams": {},
            "pid_packet_counts": {"8191": 11},
            "continuity_errors_due_to_missing_or_bad_packets": 0,
        }

    monkeypatch.setattr("oneseg.extract.record_raw_window", fake_record)
    monkeypatch.setattr("oneseg.extract.expand_raw_to_c64", fake_expand)
    monkeypatch.setattr("oneseg.extract.recording_quality", fake_quality)
    monkeypatch.setattr("oneseg.extract.analyze_capture", fake_analyze)
    monkeypatch.setattr("oneseg.extract.process_fixture", fake_deinterleave)
    monkeypatch.setattr("oneseg.extract.recover_file", fake_recover)
    return call_order


def test_extract_multiple_captures_then_close_usb_before_cpu_decode(
    tmp_path, monkeypatch
):
    tuner = FakeTuner()
    stages = _install_fake_stages(monkeypatch, tuner)
    result = extract_channel(
        tmp_path / "out", seconds=1, captures=2,
        device_factory=lambda: tuner,
    )
    assert tuner.closed
    assert stages == [
        "capture", "capture",
        "expand", "quality", "pilot", "deinterleave", "recover",
        "expand", "quality", "pilot", "deinterleave", "recover",
    ]
    assert result["gain"]["commanded_gain_db"] == -6.5
    assert result["gain"]["gain_readback_db"] == 0.0
    assert result["total_rs_verified_packets"] == 22
    assert len(result["partial_ts_files"]) == 2
    assert (tmp_path / "out/capture_01/iq.u8iq").exists()
    assert (tmp_path / "out/capture_02/deinterleaved.npz").exists()
    assert (tmp_path / "out/capture_01/partial.ts").stat().st_size == 11*188
    assert result == json.loads(
        (tmp_path / "out/extract_manifest.json").read_text()
    )
    with pytest.raises(FileExistsError):
        extract_channel(
            tmp_path / "out", device_factory=lambda: FakeTuner()
        )


def test_tmcc_failure_keeps_c64_and_pilot_fixture(tmp_path, monkeypatch):
    tuner = FakeTuner()
    stages = _install_fake_stages(
        monkeypatch, tuner, verified=False
    )
    data = extract_channel(
        tmp_path / "out", seconds=1, captures=1,
        device_factory=lambda: tuner,
    )
    assert data["windows"][0]["status"] == "tmcc_unverified"
    assert stages == ["capture", "expand", "quality", "pilot"]
    assert (tmp_path / "out/capture_01/iq.c64").exists()
    assert (tmp_path / "out/capture_01/layer_a.npz").exists()
    assert not (tmp_path / "out/capture_01/partial.ts").exists()


def test_fec_failure_keeps_deinterleaved_bits(tmp_path, monkeypatch):
    tuner = FakeTuner()
    _install_fake_stages(monkeypatch, tuner, fail_fec=True)
    data = extract_channel(
        tmp_path / "out", seconds=1, captures=1,
        device_factory=lambda: tuner,
    )
    assert data["windows"][0]["status"] == "decode_failed"
    assert "0x47" in data["windows"][0]["error"]["message"]
    assert (tmp_path / "out/capture_01/deinterleaved.npz").is_file()
    assert (tmp_path / "out/capture_01/tmcc.json").is_file()
    assert not (tmp_path / "out/capture_01/partial.ts").exists()


def test_usb_failure_keeps_successful_previous_raw_capture(tmp_path, monkeypatch):
    tuner = FakeTuner()
    _install_fake_stages(monkeypatch, tuner, fail_fec="usb")
    data = extract_channel(
        tmp_path / "out", seconds=1, captures=2,
        device_factory=lambda: tuner,
    )
    assert data["status"] == "finished_with_capture_error"
    assert data["windows"][0]["status"] == "partial_ts_extracted"
    assert data["windows"][1]["status"] == "capture_failed"
    assert (tmp_path / "out/capture_01/iq.u8iq").exists()
    assert tuner.closed


def test_invalid_settings_do_not_access_usb(tmp_path):
    touched = []
    with pytest.raises(ValueError):
        extract_channel(
            tmp_path/"bad", captures=0,
            device_factory=lambda: touched.append(1),
        )
    assert not touched
    assert not (tmp_path/"bad").exists()

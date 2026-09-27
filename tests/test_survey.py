import json

import pytest

from oneseg.survey import choose_retest_gain, run_gain_survey


def test_never_recommend_unverified_rssi_or_tmcc_only():
    rows = [
        {
            "applied_gain_db": -5.4, "successful_ts_windows": 0,
            "verified_tmcc_frames": 8, "rs_packets": 0,
            "max_fullscale_percent": 0.0,
            "median_fullscale_percent": 0.0,
        },
        {
            "applied_gain_db": -7.3, "successful_ts_windows": 2,
            "verified_tmcc_frames": 2, "rs_packets": 100,
            "max_fullscale_percent": 1.0,
            "median_fullscale_percent": 0.8,
        },
    ]
    assert choose_retest_gain(rows) == -7.3
    rows[1]["max_fullscale_percent"] = 8.0
    assert choose_retest_gain(rows) is None


def test_survey_runs_on_one_tuner_thread_with_no_native_async(
    tmp_path, monkeypatch
):
    class Device:
        valid_gains_db = [-9.9, -7.3, -5.4, 5.8]

        def __init__(self):
            self.sample_rate = None
            self.center_freq = None
            self.freq_correction = 0
            self._gain = 0
            self.closed = False

        @property
        def gain(self):
            return self._gain

        @gain.setter
        def gain(self, value):
            self._gain = min(
                self.valid_gains_db,
                key=lambda x: abs(x - value)
            )

        def read_samples_async(self, *args, **kwargs):
            raise AssertionError("native async path prohibited")

        def close(self):
            self.closed = True

    device = Device()
    counts = {"raw": 0, "probe": 0, "fec": 0}

    def fake_raw(dev, target, *, samples_required, cancelled, warmup_buffers):
        assert dev is device
        assert samples_required == 2048000
        assert warmup_buffers == 1
        counts["raw"] += 1
        target.write_bytes(bytes((126, 127)) * 64)

    def fake_expand(raw, target, *, metadata):
        assert raw.is_file() and metadata["gain_db"] == device.gain
        target.write_bytes(bytes(8))
        target.with_suffix(".c64.json").write_text("{}")
        return {}

    def fake_probe(target, *, seconds):
        assert seconds == 1
        counts["probe"] += 1
        return {
            "cp_quality": .97, "pilot_coherence": .94,
            "iq_rms": .2,
            "bch_parity_verified_frames": [{"syndrome": 0}],
        }

    def fake_decode(target, output, *, seconds, max_ofdm_symbols):
        assert seconds == 1 and max_ofdm_symbols == 1020
        counts["fec"] += 1
        return {
            "rs_and_ts_accepted_packets": 22,
            "rejected_rs_or_invalid_ts_packets": 1,
            "pat_programs": {}, "pmt_elementary_streams": {},
        }

    monkeypatch.setattr("oneseg.survey.record_raw_window", fake_raw)
    monkeypatch.setattr("oneseg.survey.expand_raw_to_c64", fake_expand)
    output = tmp_path / "auto.json"
    report = run_gain_survey(
        channel=20, seconds=1, repeats=2,
        gains=[-7.3, -5.4], output=output,
        log_directory=tmp_path,
        device_factory=lambda: device,
        probe_function=fake_probe,
        decode_function=fake_decode,
    )
    assert device.closed
    assert counts == {"raw": 4, "probe": 4, "fec": 4}
    assert report["promising_gain_for_manual_retest_db"] == -7.3
    assert len(report["windows"]) == 4
    assert output.is_file()
    assert all(w["accepted_chunk"] == 22 for w in report["windows"])
    assert all(w["checks"]["tmcc_parity"]["status"] == "pass" for w in report["windows"])
    assert any(
        json.loads(line)["event"] == "survey_complete"
        for line in open(report["log_jsonl"], encoding="utf-8")
    )
    assert not list(tmp_path.glob("*.u8iq"))


def test_gain_survey_rejects_invalid_levels_and_does_not_touch_usb(tmp_path):
    calls = []
    def fake_device():
        calls.append(1)
        raise AssertionError("never open tuner with bad settings")
    with pytest.raises(ValueError):
        run_gain_survey(gains=[1], device_factory=fake_device,
                        log_directory=tmp_path)
    assert not calls



def test_real_fc0013_zero_gain_getter_does_not_collapse_gain_groups(
    tmp_path, monkeypatch,
):
    """Regression for 2026-09-27 uploaded 14-window report.

    librtlsdr's gain getter returned 0.0 for every negative tuner step,
    although I/Q RMS and clipping varied with each accepted setter.
    """
    class ZeroGetterDevice:
        valid_gains_db = [-9.9, -7.3, 5.8]

        def __init__(self):
            self.commanded = None
            self.sample_rate = None
            self.center_freq = None
            self.freq_correction = 0
            self.closed = False

        @property
        def gain(self):
            return 0.0

        @gain.setter
        def gain(self, value):
            self.commanded = value

        def close(self):
            self.closed = True

    device = ZeroGetterDevice()

    def raw_capture(dev, target, *, samples_required, cancelled, warmup_buffers):
        assert dev is device
        assert dev.commanded in (-9.9, -7.3)
        target.write_bytes(bytes((128, 129)) * 64)

    def expand(raw, dest, *, metadata):
        assert metadata["gain_db"] == device.commanded
        dest.write_bytes(bytes(8))
        dest.with_suffix(".c64.json").write_text("{}")
        return {}

    def probe(dest, *, seconds):
        return {
            "cp_quality": .98,
            "pilot_coherence": .97,
            "iq_rms": .23,
            "bch_parity_verified_frames": [{"syndrome": 0}],
        }

    def decode(dest, output, *, seconds, max_ofdm_symbols):
        return {
            "rs_and_ts_accepted_packets": 150 if device.commanded == -9.9 else 20,
            "rejected_rs_or_invalid_ts_packets": 0,
            "pat_programs": {},
            "pmt_elementary_streams": {},
        }

    monkeypatch.setattr("oneseg.survey.record_raw_window", raw_capture)
    monkeypatch.setattr("oneseg.survey.expand_raw_to_c64", expand)
    report = run_gain_survey(
        channel=20, seconds=1, repeats=2, gains=[-9.9, -7.3],
        output=tmp_path / "summary.json", log_directory=tmp_path,
        device_factory=lambda: device,
        probe_function=probe, decode_function=decode,
    )
    assert device.closed
    assert [g["applied_gain_db"] for g in report["by_gain"]] == [-9.9, -7.3]
    assert [g["rs_packets"] for g in report["by_gain"]] == [300, 40]
    assert report["promising_gain_for_manual_retest_db"] == -9.9
    assert report["gain_readback_mismatch_count"] == 4
    assert all(x["gain_readback_db"] == 0.0 for x in report["windows"])
    assert all(not x["gain_readback_matches_command"] for x in report["windows"])
    assert all(x["applied_gain_db"] == x["gain_requested_db"]
               for x in report["windows"])

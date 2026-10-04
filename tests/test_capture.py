import json

import numpy as np
import pytest

from oneseg.capture import READ_SAMPLES, capture_channel
from oneseg.dsp import DEFAULT_SAMPLE_RATE


class FakeSdr:
    def __init__(self):
        self.calls = 0
        self.closed = False
        self.sample_rate = None
        self.center_freq = None
        self.gain = None
        self.freq_correction = 0
        self.cancelled = False

    def read_bytes(self, count):
        assert count == 2 * READ_SAMPLES
        self.calls += 1
        return bytes((100 + self.calls, 255)) * READ_SAMPLES

    def read_samples_async(self, *args, **kwargs):
        raise AssertionError("native async should never be used")

    def cancel_read_async(self):
        raise AssertionError("native async cancellation must not be used")

    def close(self):
        self.closed = True


def test_exact_capture_and_metadata_with_zero_ppm(tmp_path):
    fake = FakeSdr()
    output = tmp_path / "test.c64"
    info = capture_channel(
        output,
        channel=27,
        seconds=0.25,
        device_factory=lambda: fake,
    )
    actual = np.fromfile(output, dtype="<c8")
    assert len(actual) == round(DEFAULT_SAMPLE_RATE * 0.25)
    assert fake.calls == 1 + 4  # one warm-up plus four data reads
    assert fake.closed
    assert fake.freq_correction == 0
    assert np.allclose(actual[:10], np.complex64((102 / 127.5 - 1) + 1j))
    assert np.allclose(actual[-10:], np.complex64((105 / 127.5 - 1) + 1j))
    assert info["center_frequency_hz"] == 557_142_857
    assert info["decoding_status"] == "RAW_IQ_NOT_TS"
    assert json.loads(output.with_suffix(".c64.json").read_text())["samples"] == len(actual)


def test_refuses_to_overwrite_and_bad_channel(tmp_path):
    output = tmp_path / "existing.c64"
    output.write_bytes(b"important")
    with pytest.raises(FileExistsError):
        capture_channel(output, channel=13, device_factory=FakeSdr)
    assert output.read_bytes() == b"important"
    with pytest.raises(ValueError):
        capture_channel(tmp_path / "bad.c64", channel=12, device_factory=FakeSdr)


def test_device_closed_and_no_partial_output_after_failed_read(tmp_path):
    class BrokenSdr(FakeSdr):
        def read_bytes(self, count):
            self.calls += 1
            if self.calls == 2:
                return bytes(10)
            return bytes((100 + self.calls, 255)) * READ_SAMPLES

    fake = BrokenSdr()
    target = tmp_path / "incomplete.c64"
    with pytest.raises(IOError):
        capture_channel(target, channel=13, seconds=0.25, device_factory=lambda: fake)
    assert fake.closed
    assert not target.exists()
    assert not target.with_suffix(".c64.partial").exists()


def test_existing_sidecar_is_rejected_before_opening_tuner(tmp_path):
    target = tmp_path / "existing.c64"
    target.with_suffix(".c64.json").write_text("important metadata")

    def must_not_open():
        raise AssertionError("preflight failure must not touch USB")

    with pytest.raises(FileExistsError):
        capture_channel(target, channel=20, device_factory=must_not_open)
    assert target.with_suffix(".c64.json").read_text() == "important metadata"


def test_capture_records_commanded_gain_separately_from_bad_readback(tmp_path):
    class ZeroReadback(FakeSdr):
        valid_gains_db = [-9.9, -7.3, -5.4, 5.8]

        @property
        def gain(self):
            return 0.0

        @gain.setter
        def gain(self, value):
            self.requested_gain = value

    fake = ZeroReadback()
    metadata = capture_channel(tmp_path / "gain.c64", channel=20,
                               seconds=.25, gain=-5, device_factory=lambda: fake)
    assert fake.requested_gain == -5.4
    assert metadata["gain_db"] == -5.4
    assert metadata["gain_requested_db"] == -5
    assert metadata["gain_readback_db"] == 0
    assert not metadata["gain_readback_matches_command"]

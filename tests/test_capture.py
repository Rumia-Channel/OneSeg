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

    def read_samples_async(self, callback, num_samples):
        assert num_samples == READ_SAMPLES
        for _ in range(100):
            if self.cancelled:
                break
            self.calls += 1
            callback(np.full(
                num_samples, self.calls / 10 + 1j, dtype=np.complex64
            ), self)

    def cancel_read_async(self):
        self.cancelled = True

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
    assert np.allclose(actual[:10], np.complex64(0.2 + 1j))
    assert np.allclose(actual[-10:], np.complex64(0.5 + 1j))
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
        def read_samples_async(self, callback, num_samples):
            self.calls += 1
            callback(np.full(
                num_samples, self.calls / 10 + 1j, dtype=np.complex64
            ), self)
            self.calls += 1
            callback(np.empty(0, dtype=np.complex64), self)

    fake = BrokenSdr()
    target = tmp_path / "incomplete.c64"
    with pytest.raises(IOError):
        capture_channel(target, channel=13, seconds=0.25, device_factory=lambda: fake)
    assert fake.closed
    assert not target.exists()
    assert not target.with_suffix(".c64.partial").exists()

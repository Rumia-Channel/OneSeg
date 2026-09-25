import json

import numpy as np
import pytest

from oneseg.continuous import (
    CaptureCancelled,
    READ_SAMPLES,
    convert_u8_to_c64,
    record_stream,
)


class FakeDevice:
    def __init__(self, *, fail_at=None, short_at=None):
        self.calls = 0
        self.fail_at = fail_at
        self.short_at = short_at

    def read_bytes(self, count):
        assert count == 2 * READ_SAMPLES
        self.calls += 1
        if self.fail_at == self.calls:
            raise OSError("USB read failed")
        if self.short_at == self.calls:
            return bytes(10)
        val = min(255, 100 + self.calls)
        return bytes((val, 255)) * READ_SAMPLES

    def read_samples_async(self, *args, **kwargs):
        raise AssertionError("native asynchronous reads are disabled")

    def cancel_read_async(self):
        raise AssertionError("native async cancellation is disabled")


def test_u8_c64_scale_matches_pyrtlsdr():
    result = convert_u8_to_c64(bytes((0, 255, 128, 127)))
    assert len(result) == 2
    assert result.dtype == np.dtype("<c8")
    assert result[0].real == pytest.approx(-1, abs=1e-6)
    assert result[0].imag == pytest.approx(1, abs=1e-6)
    assert result[1].real == pytest.approx(128 / 127.5 - 1, abs=1e-6)


def test_no_async_calls_exact_count_warmup_and_metadata(tmp_path):
    device = FakeDevice()
    output = tmp_path / "fixture.c64"
    details = record_stream(
        device,
        output,
        samples_required=512000,
        metadata={"physical_channel": 20, "gain_db": 0.0},
    )
    samples = np.fromfile(output, dtype="<c8")
    assert len(samples) == 512000
    assert device.calls == 5
    assert samples[0].real == pytest.approx(102 / 127.5 - 1, abs=1e-6)
    assert samples[-1].real == pytest.approx(105 / 127.5 - 1, abs=1e-6)
    assert samples[0].imag == pytest.approx(1, abs=1e-6)
    assert details["acquisition"] == "sync_usb_reader_threaded_iq_writer"
    assert details["sample_continuity_verified"] is False
    assert json.loads(
        output.with_suffix(".c64.json").read_text(encoding="utf-8")
    ) == details
    assert not output.with_name("fixture.c64.partial").exists()


def test_short_read_does_not_publish(tmp_path):
    dev = FakeDevice(short_at=2)
    dest = tmp_path / "short.c64"
    with pytest.raises(IOError, match="short USB read"):
        record_stream(dev, dest, samples_required=512000, metadata={})
    assert not dest.exists()
    assert not dest.with_name("short.c64.partial").exists()


def test_error_and_cancellation_do_not_publish(tmp_path):
    dev = FakeDevice(fail_at=2)
    dest = tmp_path / "broken.c64"
    with pytest.raises(OSError, match="USB read failed"):
        record_stream(dev, dest, samples_required=512000, metadata={})
    assert not dest.exists()
    assert not dest.with_suffix(".c64.json").exists()

    dev = FakeDevice()
    dest = tmp_path / "cancelled.c64"
    with pytest.raises(CaptureCancelled):
        record_stream(
            dev, dest, samples_required=512000, metadata={},
            cancelled=lambda: dev.calls >= 2,
        )
    assert not dest.exists()


def test_does_not_overwrite(tmp_path):
    dev = FakeDevice()
    dest = tmp_path / "existing.c64"
    dest.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        record_stream(dev, dest, samples_required=512000, metadata={})
    assert dest.read_bytes() == b"keep"
    assert dev.calls == 0

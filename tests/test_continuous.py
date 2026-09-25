import json

import numpy as np
import pytest

from oneseg.continuous import (
    CaptureCancelled,
    READ_SAMPLES,
    record_stream,
    stream_exact_samples,
)


class AsyncDevice:
    def __init__(self, *, fail_at=None):
        self.calls = 0
        self.cancelled = False
        self.fail_at = fail_at

    def read_samples_async(self, callback, num_samples):
        assert num_samples == READ_SAMPLES
        for _ in range(20):
            if self.cancelled:
                break
            self.calls += 1
            if self.calls == self.fail_at:
                callback(np.empty(0, dtype=np.complex64), self)
            else:
                callback(
                    np.full(
                        READ_SAMPLES, self.calls / 10 + 1j,
                        dtype=np.complex64,
                    ),
                    self,
                )

    def cancel_read_async(self):
        self.cancelled = True


def test_async_record_exact_and_warmup(tmp_path):
    device = AsyncDevice()
    dest = tmp_path / "sample.c64"
    data = record_stream(
        device, dest, samples_required=512000,
        metadata={"physical_channel": 20, "gain_db": 0.0},
    )
    samples = np.fromfile(dest, dtype="<c8")
    assert len(samples) == 512000
    assert device.calls == 5  # one warmup + 4 data callbacks
    assert device.cancelled
    assert np.allclose(samples[:5], .2 + 1j)
    assert np.allclose(samples[-5:], .5 + 1j)
    assert json.loads(dest.with_suffix(".c64.json").read_text()) == data
    assert data["acquisition"] == "continuous_async"
    assert not dest.with_name("sample.c64.partial").exists()


def test_callback_failure_is_raised_and_file_not_published(tmp_path):
    device = AsyncDevice(fail_at=2)
    target = tmp_path / "bad.c64"
    with pytest.raises(IOError, match="short asynchronous"):
        record_stream(
            device, target, samples_required=512000, metadata={},
        )
    assert device.cancelled
    assert not target.exists()
    assert not target.with_name("bad.c64.partial").exists()
    assert not target.with_suffix(".c64.json").exists()


def test_cancellation_cleans_incomplete_file(tmp_path):
    device = AsyncDevice()
    target = tmp_path / "cancelled.c64"
    with pytest.raises(CaptureCancelled):
        record_stream(
            device, target, samples_required=512000,
            metadata={}, cancelled=lambda: device.calls >= 2,
        )
    assert not target.exists()
    assert device.cancelled


def test_existing_file_never_overwritten(tmp_path):
    device = AsyncDevice()
    target = tmp_path / "existing.c64"
    target.write_bytes(b"leave me alone")
    with pytest.raises(FileExistsError):
        record_stream(device, target, samples_required=100, metadata={})
    assert target.read_bytes() == b"leave me alone"
    assert device.calls == 0

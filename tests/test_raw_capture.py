import json

import numpy as np
import pytest

from oneseg.continuous import CaptureCancelled, READ_SAMPLES
from oneseg.raw_capture import record_raw_window, expand_raw_to_c64


class Device:
    def __init__(self, short_at=None):
        self.count = 0
        self.short_at = short_at

    def read_bytes(self, count):
        assert count == READ_SAMPLES * 2
        self.count += 1
        if self.count == self.short_at:
            return bytes(15)
        return bytes((min(255, 100 + self.count), 255)) * READ_SAMPLES

    def read_samples_async(self, *args, **kwargs):
        raise AssertionError("unsafe native async must never run")

    def cancel_read_async(self):
        raise AssertionError("unsafe native cancellation must never run")


def test_raw_capture_exact_bytes_and_lossless_c64_conversion(tmp_path):
    dev = Device()
    raw = tmp_path / "window.u8iq"
    size = 512000
    got = record_raw_window(
        dev, raw, samples_required=size, warmup_buffers=1
    )
    assert got == size
    assert raw.stat().st_size == size * 2
    assert dev.count == 5
    assert raw.read_bytes()[:4] == bytes((102, 255, 102, 255))
    assert raw.read_bytes()[-4:] == bytes((105, 255, 105, 255))
    c64 = tmp_path / "window.c64"
    meta = expand_raw_to_c64(
        raw, c64, metadata={"center_frequency_hz":515142857, "gain_db":0}
    )
    assert c64.stat().st_size == size * 8
    assert meta["samples"] == size
    assert meta["acquisition"] == "synchronous_raw_u8iq_direct_write"
    assert not meta["sample_continuity_verified"]
    assert json.loads(c64.with_suffix(".c64.json").read_text()) == meta
    values = np.fromfile(c64, dtype="<c8")
    assert values[0].real == pytest.approx(102 / 127.5 - 1, abs=1e-6)
    assert values[-1].real == pytest.approx(105 / 127.5 - 1, abs=1e-6)
    assert values[0].imag == pytest.approx(1, abs=1e-6)
    assert not c64.with_name(c64.name + ".partial").exists()


def test_raw_short_read_or_cancel_does_not_publish(tmp_path):
    fail = tmp_path / "bad.u8iq"
    with pytest.raises(IOError, match="short raw USB buffer"):
        record_raw_window(
            Device(short_at=2), fail, samples_required=512000
        )
    assert not fail.exists() and not (tmp_path / "bad.u8iq.partial").exists()
    device = Device()
    cancel = tmp_path / "cancel.u8iq"
    with pytest.raises(CaptureCancelled):
        record_raw_window(
            device, cancel, samples_required=512000,
            cancelled=lambda: device.count >= 1,
        )
    assert not cancel.exists()


def test_raw_and_complex_file_preexisting_are_not_overwritten(tmp_path):
    raw = tmp_path / "known.u8iq"
    raw.write_bytes(b"\x00\xff")
    with pytest.raises(FileExistsError):
        record_raw_window(Device(), raw, samples_required=1)
    output = tmp_path / "known.c64"
    output.write_bytes(b"valuable")
    with pytest.raises(FileExistsError):
        expand_raw_to_c64(raw, output, metadata={})
    assert output.read_bytes() == b"valuable"


def test_incomplete_raw_iq_pair_is_rejected(tmp_path):
    raw = tmp_path / "odd.u8iq"
    raw.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="incomplete"):
        expand_raw_to_c64(raw, tmp_path / "new.c64", metadata={})

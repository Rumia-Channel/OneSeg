import json

import numpy as np
import pytest

from oneseg.quality import block_quality, recording_quality


def test_full_scale_hit_either_i_or_q():
    iq = np.array([0.2+0.2j, 1+0j, 0.1-1j, 1-1j], dtype=np.complex64)
    hits, size, power = block_quality(iq)
    assert hits == 3
    assert size == 4
    assert power > 0


def test_recording_detects_overload_with_sidecar(tmp_path):
    iq = np.array([1+0.1j, -1+0j, 0+0.1j, 0.1+1j], dtype=np.complex64)
    path = tmp_path / "example.c64"
    iq.astype("<c8").tofile(path)
    path.with_suffix(".c64.json").write_text(json.dumps({
        "format": "complex64", "byte_order": "little-endian",
        "samples": 4, "sample_rate_hz": 2048000,
        "center_frequency_hz": 557142857,
    }))
    result = recording_quality(path)
    assert result["full_scale_fraction"] == pytest.approx(0.75)
    assert result["high_clipping_warning"]


def test_detects_truncated_capture(tmp_path):
    iq = np.array([0.1+0.2j], dtype=np.complex64)
    path = tmp_path / "truncated.c64"
    iq.astype("<c8").tofile(path)
    path.with_suffix(".c64.json").write_text(json.dumps({
        "format": "complex64", "byte_order": "little-endian", "samples": 12
    }))
    with pytest.raises(ValueError, match="mismatched"):
        recording_quality(path)

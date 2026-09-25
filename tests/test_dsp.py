import numpy as np
import pytest

from oneseg.dsp import (
    ONESEG_RATE,
    cyclic_prefix_candidates,
    power_spectrum,
    to_one_seg_rate,
)


def test_spectrum_tone():
    rate = 2_048_000
    n = np.arange(32768)
    tone_hz = 128_000
    iq = np.exp(2j * np.pi * n * tone_hz / rate).astype(np.complex64)
    f, power = power_spectrum(iq, rate, 100_000_000)
    assert abs(f[int(np.argmax(power))] - 100.128) < 0.001


def test_downsampling_rate_and_length():
    src = np.ones(20_480, dtype=np.complex64)
    out = to_one_seg_rate(src, 2_048_000)
    assert len(out) == 10_159
    assert 1_015_000 < ONESEG_RATE < 1_016_000


def test_cp_candidate_detects_known_mode():
    rng = np.random.default_rng(10)
    nfft, guard = 1024, 128
    symbols = []
    for _ in range(12):
        data = (rng.standard_normal(nfft) + 1j * rng.standard_normal(nfft)).astype(np.complex64)
        symbols.append(np.concatenate((data[-guard:], data)))
    iq = np.concatenate(symbols)
    candidates = cyclic_prefix_candidates(iq)
    # A CP of 1/8 also matches its shorter 1/16 and 1/32 subwindows;
    # a peak alone cannot unambiguously identify the true guard length.
    matching = [c for c in candidates if c["mode"] == 3 and c["guard"] == "1/8"]
    assert matching and matching[0]["correlation"] > 0.95


def test_short_capture():
    with pytest.raises(ValueError):
        cyclic_prefix_candidates(np.zeros(100, dtype=np.complex64))

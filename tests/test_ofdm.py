import numpy as np

from oneseg.dsp import ONESEG_RATE
from oneseg.ofdm import (
    central_carriers,
    extract_ofdm_symbols,
    find_symbol_lock,
)


def make_synthetic_ofdm(fft, guard, symbols=25, freq_error=140.0):
    rng = np.random.default_rng(127)
    sequence = [np.zeros(87, dtype=np.complex64)]
    reference = []
    for _ in range(symbols):
        bins = rng.choice(np.array([1+1j, 1-1j, -1+1j, -1-1j]), fft)
        waveform = np.fft.ifft(bins).astype(np.complex64)
        reference.append(bins)
        sequence.append(np.concatenate((waveform[-guard:], waveform)))
    iq = np.concatenate(sequence)
    n = np.arange(len(iq))
    iq *= np.exp(2j * np.pi * freq_error * n / ONESEG_RATE)
    return iq, np.array(reference)


def test_find_periodic_cp_and_fft_correlates_symbols():
    iq, reference = make_synthetic_ofdm(1024, 128)
    lock = find_symbol_lock(iq, modes=(3,), guards=((1, 8),))
    assert lock.mode == 3
    assert lock.guard == "1/8"
    assert lock.cp_quality > 0.95
    assert abs(lock.coarse_cfo_hz - 140) < 5
    fft = extract_ofdm_symbols(iq, lock)
    assert fft.shape[1] == 1024
    assert len(fft) >= 23
    # Per-symbol phase may differ; compare a phase-independent normalized vector.
    offset = 87
    assert abs(lock.first_sample - offset) < 2
    observed = fft[0]  # fftshift
    expected = np.fft.fftshift(reference[0])
    corr = abs(np.vdot(observed, expected)) / (
        np.linalg.norm(observed) * np.linalg.norm(expected)
    )
    assert corr > 0.99
    assert central_carriers(fft, 3).shape[1] == 325

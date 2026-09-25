"""DSP helpers; no ISDB-T video or transport-stream decoder is claimed here."""

from __future__ import annotations

import numpy as np
from scipy import signal

DEFAULT_SAMPLE_RATE = 2_048_000
ONESEG_RATE = 2_048_000 * 125 / 252
MODE_FFT = {1: 256, 2: 512, 3: 1024}
GUARD_FRACTIONS = ((1, 4), (1, 8), (1, 16), (1, 32))


def power_spectrum(iq: np.ndarray, sample_rate: int, center_hz: int, fft_size: int = 4096):
    """Average 8 Hann-windowed FFTs, returning frequency MHz and amplitude dBFS."""
    if sample_rate <= 0 or fft_size < 64 or fft_size & (fft_size - 1):
        raise ValueError("invalid rate or FFT size")
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < fft_size:
        raise ValueError("not enough samples")
    frames = min(8, len(iq) // fft_size)
    blocks = iq[-frames * fft_size :].reshape(frames, fft_size)
    window = np.hanning(fft_size).astype(np.float32)
    transforms = np.fft.fftshift(np.fft.fft(blocks * window, axis=-1), axes=-1)
    power = np.mean(np.abs(transforms) ** 2, axis=0)
    dbfs = 10 * np.log10(np.maximum(power, 1e-18)) - 20 * np.log10(np.sum(window))
    frequency_mhz = (np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / sample_rate)) + center_hz) / 1e6
    return frequency_mhz, dbfs


def to_one_seg_rate(iq: np.ndarray, source_rate: int) -> np.ndarray:
    """Resample 2.048 MS/s captures to ~1.015873 MS/s for central-segment experiments."""
    if source_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError("initial diagnostic supports only 2,048,000 samples/s")
    return signal.resample_poly(np.asarray(iq, dtype=np.complex64), 125, 252).astype(np.complex64)


def _moving_sum(values: np.ndarray, length: int) -> np.ndarray:
    summed = np.empty(len(values) + 1, dtype=np.result_type(values, np.float64))
    summed[0] = 0
    np.cumsum(values, out=summed[1:])
    return summed[length:] - summed[:-length]


def cyclic_prefix_candidates(iq: np.ndarray, limit: int = 80_000) -> list[dict]:
    """Rank CP-correlations across 1seg modes/guards; high score is NOT a decoder lock.

    Input must already be a filtered/aligned one-segment complex stream at ONESEG_RATE.
    Shorter CP windows are nested within longer true prefixes; this metric
    alone cannot uniquely identify the guard interval. Interference/noise may
    also create peaks; TMCC and FEC are still required.
    """
    iq = np.asarray(iq, dtype=np.complex64)[:limit]
    if len(iq) < 2048:
        raise ValueError("at least 2048 one-segment samples are required")
    candidates = []
    for mode, n_fft in MODE_FFT.items():
        a = iq[:-n_fft]
        b = iq[n_fft:]
        pair = np.conj(a) * b
        energy_a = np.abs(a) ** 2
        energy_b = np.abs(b) ** 2
        for numer, denom in GUARD_FRACTIONS:
            length = n_fft * numer // denom
            corr = _moving_sum(pair, length)
            norm = np.sqrt(
                np.maximum(_moving_sum(energy_a, length), 0)
                * np.maximum(_moving_sum(energy_b, length), 0)
            )
            quality = np.abs(corr) / np.maximum(norm, 1e-12)
            best = int(np.argmax(quality))
            candidates.append({
                "mode": mode,
                "guard": f"{numer}/{denom}",
                "sample": best,
                "correlation": float(quality[best]),
            })
    return sorted(candidates, key=lambda row: row["correlation"], reverse=True)

import numpy as np

from oneseg.wfm import MonoWfm


def test_wfm_audio_rate_and_finite():
    rate = 2_048_000
    t = np.arange(131_072) / rate
    deviation = 40_000
    frequency = 1_000
    phase = 2 * np.pi * deviation / frequency * (
        1 - np.cos(2 * np.pi * frequency * t)
    ) / (2 * np.pi)
    iq = np.exp(1j * phase).astype(np.complex64)
    audio = MonoWfm().process(iq)
    assert 3000 < len(audio) < 3100
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= 1

import numpy as np
import pytest

from oneseg.pilots import (
    TMCC_CARRIERS,
    TMCC_SYNC_EVEN,
    TMCC_SYNC_ODD,
    candidate_tmcc_sync,
    central_pilot_polarities,
    equalize_segment,
    find_pilot_alignment,
    scattered_pilot_indices,
    tmcc_differential_soft_bits,
)


def synthetic_fft(shift=12, phase=2, symbols=40):
    rng = np.random.default_rng(53)
    fft = np.zeros((symbols, 1024), dtype=np.complex64)
    pol = central_pilot_polarities()
    for row in range(symbols):
        carrier = rng.choice(
            np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64),
            size=433,
        )
        pilot_positions = scattered_pilot_indices(row, phase)
        carrier[pilot_positions] = pol[pilot_positions]
        carrier[TMCC_CARRIERS] = 1.33
        # Unknown channel phase slope across frequency:
        channel = np.exp(1j * np.arange(433) * 0.017)
        fft[row, 512 - 216 + shift + np.arange(433)] = carrier * channel
    return fft


def test_true_integer_carrier_shift_and_unknown_symbol_phase():
    fft = synthetic_fft()
    found = find_pilot_alignment(fft, max_shift=20)
    assert found.integer_offset_bins == 12
    assert found.symbol_phase == 2
    assert found.pilot_coherence > 0.98


def test_equalized_pilots_have_expected_polarity():
    fft = synthetic_fft(shift=-7, phase=1)
    found = find_pilot_alignment(fft, max_shift=16)
    equalized = equalize_segment(fft, found)
    expected = central_pilot_polarities()
    for row in range(10):
        selected = scattered_pilot_indices(row, found.symbol_phase)
        np.testing.assert_allclose(
            equalized[row, selected], expected[selected], atol=1e-4
        )


def test_tmcc_polarity_differential_soft_decisions():
    eq = np.ones((20, 433), dtype=np.complex64)
    levels = np.array([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1])
    eq[:, TMCC_CARRIERS] = levels[:, None]
    soft = tmcc_differential_soft_bits(eq)
    expected = levels[1:] * levels[:-1]
    np.testing.assert_array_equal(soft, expected)


def test_repeated_tmcc_sync_remains_unverified_without_bch():
    bits = np.zeros(500, dtype=np.uint8)
    bits[45:61] = TMCC_SYNC_EVEN
    bits[249:265] = TMCC_SYNC_ODD
    soft = (1 - 2 * bits.astype(np.float32))
    found = candidate_tmcc_sync(soft, max_errors=0)
    assert any(p["bit_index"] == 45 for p in found["repeated_sync_candidates"])
    assert found["tmcc_bch_verified"] is False
    assert found["mpeg_ts_recovered"] is False


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        find_pilot_alignment(np.zeros((5, 1024), dtype=np.complex64))
    with pytest.raises(ValueError):
        equalize_segment(np.zeros((4, 433)), find_pilot_alignment(synthetic_fft()))

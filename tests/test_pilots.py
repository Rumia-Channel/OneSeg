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



def test_batched_equalization_matches_original_linear_interpolation():
    fft = synthetic_fft(shift=12, phase=2, symbols=200)
    alignment = find_pilot_alignment(fft, max_shift=16)
    batch = equalize_segment(fft, alignment)
    expected = np.empty_like(batch)
    original = fft[
        :, 512 - 216 + alignment.integer_offset_bins
           + np.arange(433)
    ]
    reference = central_pilot_polarities()
    for row in range(len(fft)):
        positions = scattered_pilot_indices(row, alignment.symbol_phase)
        h = original[row, positions] / reference[positions]
        interp = (
            np.interp(np.arange(433), positions, h.real)
            + 1j * np.interp(np.arange(433), positions, h.imag)
        )
        expected[row] = original[row] / interp
    np.testing.assert_allclose(batch, expected, rtol=2e-5, atol=2e-5)



def test_individual_tmcc_carrier_recovers_parity_when_three_other_carriers_fail():
    from oneseg.pilots import (
        TMCC_SYNC_EVEN, TMCC_SYNC_ODD, TMCC_CARRIERS,
        select_tmcc_soft_stream,
    )
    from oneseg.tmcc import _GENERATOR

    def encoded_frame(sync):
        frame = np.zeros(204, dtype=np.uint8)
        frame[1:17] = sync
        frame[27] = 1
        frame[28:41] = [
            0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1
        ]  # QPSK 2/3 I=4 one segment
        code = 0
        for bit in frame[20:122]:
            code = (code << 1) | int(bit)
        remainder = code << 82
        for degree in range(183, 81, -1):
            if remainder & (1 << degree):
                remainder ^= _GENERATOR << (degree - 82)
        frame[122:] = [
            (remainder >> shift) & 1 for shift in range(81, -1, -1)
        ]
        return frame

    bits = np.concatenate([
        encoded_frame(TMCC_SYNC_EVEN),
        encoded_frame(TMCC_SYNC_ODD),
        encoded_frame(TMCC_SYNC_EVEN),
    ])
    # Original source is a coherent carrier with known DBPSK symbols.
    polarity = np.cumprod(
        (1 - 2 * bits.astype(np.int8)).astype(np.int8)
    ).astype(np.float32)
    eq = np.ones((len(bits) + 1, 433), dtype=np.complex64)
    eq[1:, TMCC_CARRIERS[0]] = polarity
    rng = np.random.default_rng(823)
    for carrier in TMCC_CARRIERS[1:]:
        eq[1:, carrier] = rng.choice(
            np.array([-1, 1], dtype=np.float32), len(bits)
        )
    _, report = select_tmcc_soft_stream(eq)
    assert report["tmcc_bch_verified"]
    assert report["tmcc_soft_source"] == f"carrier_{int(TMCC_CARRIERS[0])}"
    assert len(report["bch_parity_verified_frames"]) == 3
    assert report["tmcc_repeated_verified_pairs"] == 2
    assert report["bch_parity_verified_frames"][0]["layer_A"]["segments"] == 1


def test_tmcc_source_keeps_combined_reference_on_equal_verified_counts():
    from oneseg.pilots import (
        TMCC_CARRIERS, select_tmcc_soft_stream
    )
    eq = np.ones((300, 433), dtype=np.complex64)
    _, report = select_tmcc_soft_stream(eq)
    assert not report["tmcc_bch_verified"]
    assert report["tmcc_soft_source"] == "combined"
    assert set(report["tmcc_per_carrier_verified_counts"]) == {
        "combined", *(f"carrier_{int(i)}" for i in TMCC_CARRIERS)
    }

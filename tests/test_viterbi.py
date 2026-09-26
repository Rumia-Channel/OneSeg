import numpy as np
import pytest

from oneseg.viterbi import (
    PUNCTURE,
    convolutional_encode_test,
    decode_soft_bits,
    depuncture,
    _fast_trellis,
    _transitions,
)


@pytest.mark.parametrize("rate", list(PUNCTURE))
def test_roundtrip_all_isdbt_rates(rate):
    rng = np.random.default_rng(200)
    # Number of input bits must be a multiple of each puncturing step.
    size = len(PUNCTURE[rate]) * 24
    bits = rng.integers(0, 2, size=size, dtype=np.uint8)
    bits[-6:] = 0
    encoded = convolutional_encode_test(bits, rate)
    decoded = decode_soft_bits(encoded.astype(np.float32), rate, end_state_zero=True)
    np.testing.assert_array_equal(decoded, bits)


def test_corrects_corrupted_soft_inputs():
    bits = np.random.default_rng(5).integers(0, 2, size=256, dtype=np.uint8)
    bits[-6:] = 0
    soft = convolutional_encode_test(bits).astype(np.float32)
    soft[[55, 89, 220, 380]] = 1 - soft[[55, 89, 220, 380]]
    decoded = decode_soft_bits(soft, end_state_zero=True)
    np.testing.assert_array_equal(decoded, bits)


def test_erasure_and_length_validation():
    with pytest.raises(ValueError):
        depuncture(np.array([0, 1, 1, 0], np.float32), "2/3")
    with pytest.raises(ValueError):
        depuncture(np.array([-0.1, 1], np.float32), "1/2")



def test_numba_trellis_matches_reference_on_noisy_midstream_soft_symbols():
    # Roundtrip with a clean encoded test vector does not detect subtle
    # numeric/tie-breaking regressions in an actual noisy RF stream.
    rng = np.random.default_rng(162)
    for rate in ("1/2", "2/3", "3/4"):
        mask = PUNCTURE[rate]
        received = rng.uniform(
            low=0, high=1,
            size=sum(mask) * 91,
        ).astype(np.float32)
        symbols = depuncture(received, rate)
        pred0, pred1, expected = _transitions()
        scores = np.full(64, np.inf, dtype=np.float64)
        scores[0] = 0
        history = np.empty((len(symbols), 64), dtype=np.uint8)
        for step, pair in enumerate(symbols):
            a = scores[pred0] + np.sum(
                np.abs(expected[0] - pair[None, :]), axis=1
            )
            b = scores[pred1] + np.sum(
                np.abs(expected[1] - pair[None, :]), axis=1
            )
            chosen = b < a
            history[step] = chosen
            scores = np.where(chosen, b, a)
            scores -= np.min(scores)
        state = int(np.argmin(scores))
        reference = np.empty(len(symbols), dtype=np.uint8)
        for step in range(len(symbols)-1, -1, -1):
            reference[step] = state & 1
            state = (state >> 1) | (int(history[step, state]) << 5)
        np.testing.assert_array_equal(
            decode_soft_bits(received, rate), reference
        )
    assert _fast_trellis.signatures

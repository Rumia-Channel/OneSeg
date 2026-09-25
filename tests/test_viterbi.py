import numpy as np
import pytest

from oneseg.viterbi import (
    PUNCTURE,
    convolutional_encode_test,
    decode_soft_bits,
    depuncture,
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

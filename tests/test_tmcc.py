import numpy as np
import pytest
from oneseg.tmcc import (
    _GENERATOR,
    MODULATION,
    CODE_RATE,
    tmcc_syndrome,
    parse_tmcc_frame,
    verify_frames_from_soft,
)
from oneseg.pilots import TMCC_SYNC_EVEN, TMCC_SYNC_ODD


def test_reference_polynomial_is_degree_82_with_required_terms():
    assert _GENERATOR.bit_length() == 83
    assert _GENERATOR & 1
    assert _GENERATOR & (1 << 77)
    assert not (_GENERATOR & (1 << 81))


def make_frame(sync):
    frame = np.zeros(204, dtype=np.uint8)
    frame[1:17] = sync
    # A: QPSK (001), code 1/2 (000), I=2 Mode3 (010), one segment (0001)
    frame[28:41] = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # B: 64QAM / 3/4 / I=0 / 12 segments
    frame[41:54] = [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    # layer C unused
    frame[54:67] = [1] * 13
    frame[27] = 1
    # Encode info polynomial systematically, independent bitwise long division.
    info = frame[20:122]
    val = 0
    for bit in info:
        val = (val << 1) | int(bit)
    val <<= 82
    remainder = val
    for degree in range(183, 81, -1):
        if remainder & (1 << degree):
            remainder ^= _GENERATOR << (degree - 82)
    parity = [((remainder >> shift) & 1) for shift in range(81, -1, -1)]
    frame[122:204] = parity
    return frame


def test_encoded_nonzero_tmcc_frame_validates_and_parses():
    frame = make_frame(TMCC_SYNC_EVEN)
    assert tmcc_syndrome(frame) == 0
    fields = parse_tmcc_frame(frame)
    assert fields["bch_parity_verified"]
    assert not fields["bit_error_correction_performed"]
    assert fields["layer_A"]["modulation"] == "QPSK"
    assert fields["layer_A"]["code_rate"] == "1/2"
    assert fields["layer_A"]["time_interleaving_mode3"] == 2
    assert fields["layer_A"]["segments"] == 1
    assert fields["layer_B"]["segments"] == 12
    assert fields["layer_C"]["segments"] == "UNUSED"
    assert fields["partial_reception_flag"] is True


def test_single_info_or_parity_error_is_rejected():
    frame = make_frame(TMCC_SYNC_ODD)
    for position in (20, 40, 101, 122, 203):
        corrupt = frame.copy()
        corrupt[position] ^= 1
        assert tmcc_syndrome(corrupt) != 0
        with pytest.raises(ValueError, match="parity check failed"):
            parse_tmcc_frame(corrupt)


def test_verified_pair_204_symbols_apart():
    frames = [make_frame(TMCC_SYNC_EVEN), make_frame(TMCC_SYNC_ODD)]
    bits = np.concatenate([*frames, np.tile([0, 1], 40)]).astype(np.uint8)
    soft = 1.0 - 2.0 * bits.astype(np.float32)
    result = verify_frames_from_soft(soft)
    accepted = result["bch_parity_verified_frames"]
    assert result["tmcc_bch_verified"]
    assert any(x["frame_start_bit_index"] == 0 for x in accepted)
    assert any(x["frame_start_bit_index"] == 204 for x in accepted)
    assert all(x["syndrome"] == 0 for x in accepted)
    assert not result["mpeg_ts_recovered"]


def test_repeated_sync_alone_not_tmcc_lock():
    frames = [make_frame(TMCC_SYNC_EVEN), make_frame(TMCC_SYNC_ODD)]
    frames[0][50] ^= 1
    frames[1][180] ^= 1
    bits = np.concatenate(frames)
    results = verify_frames_from_soft(1.0 - 2.0 * bits.astype(np.float32))
    assert not results["tmcc_bch_verified"]
    assert results["bch_parity_verified_frames"] == []


def test_wrong_lengths_or_nonbinary_rejected():
    with pytest.raises(ValueError):
        tmcc_syndrome(np.zeros(184, dtype=np.uint8))
    frame = np.zeros(204, dtype=np.uint8)
    frame[20] = 2
    with pytest.raises(ValueError):
        tmcc_syndrome(frame)
    with pytest.raises(ValueError):
        verify_frames_from_soft(np.zeros(300), max_sync_errors=-1)

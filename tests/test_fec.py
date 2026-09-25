import pytest

from oneseg.fec import OuterReedSolomon, prbs_bytes


def test_rs_204_188_roundtrip_and_byte_correction():
    rs = OuterReedSolomon()
    payload = bytes([0x47]) + bytes(range(187))
    frame = rs.encode_test_vector(payload)
    assert len(frame) == 204
    damaged = bytearray(frame)
    for i in (2, 7, 12, 40, 180, 198):
        damaged[i] ^= 0x5A
    assert rs.decode(bytes(damaged)) == payload


def test_rs_rejects_short_and_uncorrectable():
    rs = OuterReedSolomon()
    with pytest.raises(ValueError):
        rs.decode(b"hello")
    raw = bytearray(rs.encode_test_vector(bytes(range(188))))
    for i in range(20):
        raw[i] ^= (i + 1)
    with pytest.raises(ValueError):
        rs.decode(raw)


def test_prbs_deterministic_and_byte_xor_symmetry():
    mask = prbs_bytes(203)
    assert len(mask) == 203
    assert mask == prbs_bytes(203)
    assert mask != bytes(203)
    data = bytes(range(203))
    scrambled = bytes(a ^ b for a, b in zip(data, mask))
    assert bytes(a ^ b for a, b in zip(scrambled, mask)) == data

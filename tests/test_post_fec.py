import pytest

from oneseg.fec import OuterReedSolomon
from oneseg.post_fec import ByteDeinterleaver, EnergyDescrambler, PostViterbiTransport


def packet():
    return bytes.fromhex("471fff10") + b"\xff" * 184


def inverse_energy_block(energy, rs):
    # Build an encoded packet with synthetic FEC then invert the exact
    # descrambler packet mapping for a deterministic unit fixture.
    result = bytearray(204)
    result[-1] = rs[0]
    for i, value in enumerate(rs[1:]):
        result[i] = value ^ energy._clock()
    energy._clock()
    return bytes(result)


def test_rs_energy_pipeline_with_partial_stream_chunks():
    original = packet()
    rs = OuterReedSolomon()
    energy = EnergyDescrambler()
    wire = inverse_energy_block(energy, rs.encode_test_vector(original))
    decoder = PostViterbiTransport(byte_deinterleaving=False)
    assert decoder.feed(wire[:83], frame_begin=True) == []
    assert decoder.feed(wire[83:]) == [original]
    assert decoder.stats.good_ts_packets == 1


def test_byte_deinterleaver_branch_delays_and_reset():
    deint = ByteDeinterleaver()
    data = bytes(range(240))
    out = deint.feed(data)
    # Branch 11 is zero-delay; branch 0 has 187 branch samples delay.
    assert out[11] == data[11]
    assert out[0] == 0
    deint.reset()
    assert deint.feed(data) == out


def test_frame_begin_requires_packet_boundary():
    post = PostViterbiTransport(byte_deinterleaving=False)
    post.feed(b"data")
    with pytest.raises(ValueError):
        post.feed(b"rest", frame_begin=True)

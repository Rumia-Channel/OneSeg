import numpy as np
import pytest

from oneseg.fec import OuterReedSolomon, prbs_bytes
from oneseg.recover import (
    FRAME_PACKETS,
    PACKET_SIZE,
    _unmask_blocks,
    find_energy_phase,
    recover_verified_packets,
    sync_byte_phase,
    byte_deinterleaved_blocks,
)


def synthetic_scrambled_blocks(count=44, phase=56):
    mask = np.frombuffer(
        prbs_bytes(FRAME_PACKETS * PACKET_SIZE), dtype=np.uint8
    ).reshape(FRAME_PACKETS, PACKET_SIZE)
    rs = OuterReedSolomon()
    rows = np.empty((count, PACKET_SIZE), dtype=np.uint8)
    for j in range(count):
        packet = bytes.fromhex("471fff10") + bytes([j % 256]) * 184
        code = rs.encode_test_vector(packet)
        row = np.frombuffer(code, dtype=np.uint8).copy()
        row[1:] ^= mask[(j + phase) % 64, :203]
        rows[j] = row
    return rows


def test_sync_phase_requires_repeating_204_byte_pattern():
    samples = np.zeros(204 * 64, dtype=np.uint8)
    samples[139::204] = 0x47
    assert sync_byte_phase(samples) == (139, 64)
    with pytest.raises(ValueError, match="no reliable"):
        sync_byte_phase(np.zeros(204*64, dtype=np.uint8))


def test_prbs_phase_is_selected_by_real_rs_parity():
    blocks = synthetic_scrambled_blocks()
    phase, hits = find_energy_phase(blocks, probe_packets=25)
    assert phase == 56
    assert hits == 25
    recovered, rejected = recover_verified_packets(blocks, phase)
    assert len(recovered) == len(blocks)-11
    assert rejected == 0
    assert all(packet[:4] == bytes.fromhex("471fff10")
               for _, packet in recovered)


def test_wrong_prbs_phase_cannot_pass_rs():
    blocks = synthetic_scrambled_blocks()
    true, _ = find_energy_phase(blocks, probe_packets=20)
    assert true == 56
    recovered, rejected = recover_verified_packets(blocks, 3)
    assert not recovered
    assert rejected == len(blocks)-11


def test_byte_deinterleave_preserves_sync_only_at_selected_position():
    # As in the real fixture: raw 0x47 every 204 bytes at position 139.
    raw = np.zeros(204*40, dtype=np.uint8)
    raw[139::204] = 0x47
    blocks = byte_deinterleaved_blocks(raw, 139)
    assert np.all(blocks[12:, 0] == 0x47)
    with pytest.raises(ValueError):
        byte_deinterleaved_blocks(raw, 204)

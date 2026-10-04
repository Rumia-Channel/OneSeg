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


def minimal_recovery_fixture(tmp_path):
    import json
    path = tmp_path / "bits.npz"
    bits = 2 * (384 * 221 - 120)
    np.savez(path, qpsk_soft_metric_proxy=np.ones(bits, dtype=np.float32))
    metadata = {
        "mode": 3, "partial_reception": True, "modulation": "QPSK",
        "code_rate": "2/3", "segments": 1,
        "output_data_symbol_rows": 221, "source_fft_symbols": 601,
        "time_interleaving_parameter": 4, "time_deinterleaver_warmup_symbols": 380,
        "qpsk_bit_delay_carriers": 120, "output_bit_count": bits,
        "verified_tmcc_frame_starts_original_fft_rows": [17, 221],
    }
    path.with_suffix(".npz.json").write_text(json.dumps(metadata))
    return path, metadata


@pytest.mark.parametrize("field,value,message", [
    ("partial_reception", False, "requires Mode-3"),
    ("segments", True, "requires Mode-3"),
    ("output_data_symbol_rows", 221.0, "must be integers"),
    ("source_fft_symbols", 600, "metadata disagree"),
    ("time_deinterleaver_warmup_symbols", 0, "metadata disagree"),
    ("qpsk_bit_delay_carriers", 0, "metadata disagree"),
    ("output_bit_count", 1, "metadata disagree"),
    ("verified_tmcc_frame_starts_original_fft_rows", [True], "validated TMCC"),
    ("verified_tmcc_frame_starts_original_fft_rows", [221, 17], "validated TMCC"),
    ("verified_tmcc_frame_starts_original_fft_rows", [17, 17], "validated TMCC"),
    ("verified_tmcc_frame_starts_original_fft_rows", [400], "validated TMCC"),
])
def test_recovery_rejects_inconsistent_provenance(tmp_path, field, value, message):
    import json
    from oneseg.recover import recover_file
    path, metadata = minimal_recovery_fixture(tmp_path)
    metadata[field] = value
    path.with_suffix(".npz.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        recover_file(path, tmp_path / "never.ts")
    assert not (tmp_path / "never.ts").exists()


@pytest.mark.parametrize("metrics", [np.array(0.1), np.ones((1, 169488)),
                                    np.ones(169488, dtype=np.complex64)])
def test_recovery_rejects_bad_metric_shape_and_type(tmp_path, metrics):
    from oneseg.recover import recover_file
    path, _ = minimal_recovery_fixture(tmp_path)
    np.savez(path, qpsk_soft_metric_proxy=metrics)
    with pytest.raises(ValueError, match="QPSK"):
        recover_file(path, tmp_path / "never.ts")


@pytest.mark.parametrize("collision", ["partial", "sidecar", "output"])
def test_recovery_preserves_racing_output_files(tmp_path, monkeypatch, collision):
    from pathlib import Path
    import oneseg.recover as module

    fixture, _ = minimal_recovery_fixture(tmp_path)
    packet = bytes.fromhex("471fff10") + bytes(184)
    monkeypatch.setattr(module, "decode_soft_bits", lambda *args: np.zeros(24, np.uint8))
    monkeypatch.setattr(module, "sync_byte_phase", lambda *args: (0, 64))
    monkeypatch.setattr(module, "byte_deinterleaved_blocks", lambda *args: np.zeros((20, 204), np.uint8))
    monkeypatch.setattr(module, "find_energy_phase", lambda *args: (0, 3))
    monkeypatch.setattr(module, "recover_verified_packets", lambda *args: ([(i, packet) for i in range(11, 14)], 0))
    output = tmp_path / "out.ts"
    partial = tmp_path / "out.ts.partial"
    sidecar = tmp_path / "out.ts.json"
    target = {"partial": partial, "sidecar": sidecar, "output": output}[collision]
    original_open = Path.open
    original_publish = module.publish_new_file

    def race_open(path, mode="r", *args, **kwargs):
        if path == target and mode in ("xb", "x"):
            with original_open(path, "wb") as handle:
                handle.write(b"concurrent owner")
        return original_open(path, mode, *args, **kwargs)

    def race_publish(source_path, target_path):
        with original_open(target_path, "wb") as handle:
            handle.write(b"concurrent owner")
        return original_publish(source_path, target_path)

    monkeypatch.setattr(Path, "open", race_open)
    if collision == "output":
        monkeypatch.setattr(module, "publish_new_file", race_publish)
    with pytest.raises(FileExistsError):
        module.recover_file(fixture, output)
    assert target.read_bytes() == b"concurrent owner"
    assert all(not path.exists() for path in (output, partial, sidecar) if path != target)

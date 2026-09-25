from oneseg.fec import OuterReedSolomon
from oneseg.ts_cli import decode_outer_file, inspect_ts


def packet():
    return bytes.fromhex("471fff10") + b"\xff" * 184


def test_shortened_rs_codewords_to_ts(tmp_path):
    src = tmp_path / "source.rs204"
    target = tmp_path / "decoded.ts"
    original = packet() * 4
    rs = OuterReedSolomon()
    src.write_bytes(b"".join(rs.encode_test_vector(original[n:n + 188]) for n in range(0, len(original), 188)))
    count, rejected = decode_outer_file(src, target)
    assert count == 4 and rejected == 0
    assert target.read_bytes() == original
    stats, framing = inspect_ts(target)
    assert stats.packets == 4 and framing.dropped_bytes == 0

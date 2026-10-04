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

def test_external_post_viterbi_stream_to_ts(tmp_path):
    from oneseg.post_fec import EnergyDescrambler
    from oneseg.ts_cli import decode_post_fec_file
    original = packet()
    rs = OuterReedSolomon()
    mask = EnergyDescrambler()
    coded = rs.encode_test_vector(original)
    wire = bytes(coded[i + 1] ^ mask._clock() for i in range(203)) + coded[:1]
    source = tmp_path / "postfec.bin"
    target = tmp_path / "postfec.ts"
    source.write_bytes(wire)
    accepted, rejected = decode_post_fec_file(
        source, target, byte_deinterleaving=False
    )
    assert accepted == 1 and rejected == 0
    assert target.read_bytes() == original


def test_conversion_never_overwrites_source_or_existing_output(tmp_path):
    import pytest
    from oneseg.ts_cli import decode_post_fec_file
    source = tmp_path / "input.rs204"
    original = OuterReedSolomon().encode_test_vector(packet())
    source.write_bytes(original)
    for convert in (decode_outer_file, decode_post_fec_file):
        with pytest.raises(FileExistsError):
            convert(source, source)
        assert source.read_bytes() == original
        target = tmp_path / "existing.ts"
        target.write_bytes(b"valuable")
        with pytest.raises(FileExistsError):
            convert(source, target)
        assert target.read_bytes() == b"valuable"


def test_truncated_conversion_does_not_publish_partial_success(tmp_path):
    import pytest
    from oneseg.ts_cli import decode_post_fec_file
    source = tmp_path / "input.bin"
    source.write_bytes(OuterReedSolomon().encode_test_vector(packet()) + b"bad")
    for convert in (decode_outer_file, decode_post_fec_file):
        target = tmp_path / "failed.ts"
        with pytest.raises(ValueError, match="partial"):
            convert(source, target)
        assert not target.exists()
        assert not list(tmp_path.glob(".*.partial"))


def test_output_race_does_not_replace_existing_file(tmp_path, monkeypatch):
    import pytest
    import oneseg.ts_cli as cli
    source = tmp_path / "input.bin"
    source.write_bytes(OuterReedSolomon().encode_test_vector(packet()))
    target = tmp_path / "race.ts"
    import os
    from oneseg import file_output
    operation = "rename" if os.name == "nt" else "link"
    link = getattr(file_output.os, operation)

    def create_concurrent_output(src, dst):
        target.write_bytes(b"concurrent capture")
        link(src, dst)

    monkeypatch.setattr(file_output.os, operation, create_concurrent_output)
    with pytest.raises(FileExistsError):
        cli.decode_outer_file(source, target)
    assert target.read_bytes() == b"concurrent capture"
    assert not list(tmp_path.glob(".*.partial"))

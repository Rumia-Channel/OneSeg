import pytest

from oneseg.ts import (
    TransportFramer,
    TransportStats,
    TransportWriter,
    mpeg_crc32,
    parse_packet,
    pat_programs,
    pmt_streams,
)


def packet(pid, cc, payload=b"", start=False):
    payload = payload[:184]
    hdr = bytes((0x47, (0x40 if start else 0) | (pid >> 8), pid & 255, 0x10 | cc))
    return hdr + payload + b"\xff" * (184 - len(payload))


def section_packet(table_id, pid, contents):
    section_length = len(contents) + 4
    section = bytes((table_id, 0xB0 | (section_length >> 8), section_length & 255)) + contents
    crc = mpeg_crc32(section).to_bytes(4, "big")
    return packet(pid, 0, bytes((0,)) + section + crc, True)


def test_crc_known_mpeg2():
    assert mpeg_crc32(b"123456789") == 0x0376E6E7


def test_ts_framer_partial_reads_and_resync():
    p1, p2 = packet(99, 1), packet(99, 2)
    framer = TransportFramer()
    assert framer.feed(b"bad" + p1[:75]) == []
    assert framer.feed(p1[75:] + p2[:20]) == [p1]
    assert framer.feed(p2[20:]) == [p2]
    assert framer.packets == 2
    assert framer.dropped_bytes == 3


def test_packet_rejects_error_flag_and_invalid_af():
    p = packet(40, 0)
    with pytest.raises(ValueError):
        parse_packet(bytes([0x47, p[1] | 0x80]) + p[2:])
    with pytest.raises(ValueError):
        parse_packet(p[:3] + b"\x00" + p[4:])


def test_pat_pmt_and_stats():
    # PAT: ts_id, version, section number, last section, program 7 -> pid 0x0100
    pat = section_packet(0, 0, bytes.fromhex("0001c100000007e100"))
    # PMT: program 7, version, section no, PCR PID=0x0200, program info 0,
    # video AVC PID 0x0200; AAC PID 0x0201
    pmt = section_packet(2, 0x100, bytes.fromhex(
        "0007c10000e200f0001be200f0000fe201f000"
    ))
    assert pat_programs(pat) == {7: 0x100}
    assert pmt_streams(pmt, 0x100) == {0x200: 0x1B, 0x201: 0x0F}
    stats = TransportStats()
    stats.accept(pat)
    stats.accept(pmt)
    stats.accept(packet(0x200, 4))
    stats.accept(packet(0x200, 6))
    assert stats.elementary_streams == {0x200: 0x1B, 0x201: 0x0F}
    assert stats.continuity_errors == 1


def test_writer_only_accepts_ts(tmp_path):
    path = tmp_path / "demo.ts"
    p = packet(0x1FFF, 0)
    with TransportWriter(path) as w:
        w.write(p)
        with pytest.raises(ValueError):
            w.write(bytes(188))
        assert w.stats.packets == 1
    assert path.read_bytes() == p

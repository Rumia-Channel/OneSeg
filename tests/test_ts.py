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


def psi_section(table_id, contents):
    size = len(contents) + 4
    header = bytes((table_id, 0xB0 | (size >> 8), size & 255))
    body = header + contents
    return body + mpeg_crc32(body).to_bytes(4, "big")


def pat_section(programs, *, version=0, number=0, last=0, current=True, tsid=1):
    body = tsid.to_bytes(2, "big") + bytes((0xC0 | version << 1 | current, number, last))
    for program, pid in programs.items():
        body += program.to_bytes(2, "big") + (0xE000 | pid).to_bytes(2, "big")
    return psi_section(0, body)


def pmt_section(streams, *, program=7, version=0, current=True, descriptors=b""):
    body = program.to_bytes(2, "big") + bytes((0xC0 | version << 1 | current, 0, 0))
    body += bytes.fromhex("e200") + (0xF000 | len(descriptors)).to_bytes(2, "big")
    body += descriptors
    for pid, stream_type in streams.items():
        body += bytes((stream_type,)) + (0xE000 | pid).to_bytes(2, "big") + b"\xf0\x00"
    return psi_section(2, body)


def payload_packet(pid, cc, payload=b"", *, start=False, discontinuity=False, pcr=None):
    """Adaptation stuffing leaves exactly the given payload, including no payload."""
    assert len(payload) <= 182
    control = 0x30 if payload else 0x20
    flags = (0x80 if discontinuity else 0) | (0x10 if pcr is not None else 0)
    adaptation = bytes((flags,)) + (pcr or b"")
    assert len(adaptation) <= 183 - len(payload)
    adaptation += b"\xff" * (183 - len(payload) - len(adaptation))
    header = bytes((0x47, (0x40 if start else 0) | pid >> 8, pid & 255, control | cc))
    return header + bytes((len(adaptation),)) + adaptation + payload


def section_packets(pid, section, *, cc=0, first_size=183):
    first = section[:first_size]
    if len(first) == 183:
        packets = [packet(pid, cc, b"\x00" + first, True)]
    else:
        packets = [payload_packet(pid, cc, b"\x00" + first, start=True)]
    rest = section[first_size:]
    while rest:
        cc = (cc + 1) & 15
        packets.append(packet(pid, cc, rest[:184]))
        rest = rest[184:]
    return packets


def accept_section(stats, pid, section, *, cc=0):
    for raw in section_packets(pid, section, cc=cc):
        stats.accept(raw)


def test_incremental_pat_pmt_across_many_packets():
    programs = {n: 0x100 + n for n in range(1, 101)}
    streams = {0x400 + n: 0x1B if n == 0 else 0x0F for n in range(70)}
    stats = TransportStats()
    pats = section_packets(0, pat_section(programs))
    assert len(pats) == 3
    for raw in pats[:-1]:
        stats.accept(raw)
        assert stats.pat == {}
    stats.accept(pats[-1])
    assert stats.pat == programs
    pmts = section_packets(0x107, pmt_section(streams))
    assert len(pmts) == 2
    stats.accept(pmts[0])
    assert stats.elementary_streams == {}
    stats.accept(pmts[1])
    assert stats.elementary_streams == streams
    assert stats.continuity_errors == 0
    assert pat_programs(pats[0]) == {}
    assert pmt_streams(pmts[0], 0x107) == {}


@pytest.mark.parametrize("first_size", [1, 2, 3, 8, 12])
def test_split_psi_header_and_body(first_size):
    stats = TransportStats()
    for raw in section_packets(0, pat_section({7: 0x100}), first_size=first_size):
        stats.accept(raw)
    assert stats.pat == {7: 0x100}


def test_pointer_finishes_previous_section_and_starts_next():
    stats = TransportStats()
    first = pat_section({7: 0x100}, number=0, last=1)
    second = pat_section({8: 0x101}, number=1, last=1)
    stats.accept(payload_packet(0, 0, b"\x00" + first[:10], start=True))
    remainder = first[10:]
    stats.accept(packet(0, 1, bytes((len(remainder),)) + remainder + second, True))
    assert stats.pat == {7: 0x100, 8: 0x101}


def test_multiple_sections_and_pointer_stuffing_in_single_packet():
    first = pat_section({7: 0x100}, number=0, last=1)
    second = pat_section({8: 0x101}, number=1, last=1)
    raw = packet(0, 0, b"\x03\xff\xff\xff" + first + second, True)
    assert pat_programs(raw) == {7: 0x100, 8: 0x101}
    stats = TransportStats()
    stats.accept(raw)
    assert stats.pat == {7: 0x100, 8: 0x101}


def test_multi_section_pat_atomic_replacement_and_version_wrap():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100, 9: 0x103}, version=31))
    accept_section(stats, 0x100, pmt_section({0x200: 0x1B}))
    accept_section(stats, 0, pat_section({8: 0x101}, number=1, last=1), cc=1)
    assert stats.pat == {7: 0x100, 9: 0x103}
    # A repeated active version must not discard a pending next version.
    accept_section(stats, 0, pat_section({7: 0x100, 9: 0x103}, version=31), cc=2)
    accept_section(stats, 0, pat_section({7: 0x100}, number=0, last=1), cc=3)
    assert stats.pat == {7: 0x100, 8: 0x101}
    assert stats.elementary_streams == {0x200: 0x1B}


def test_different_pat_versions_never_combine_sections():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100}, number=0, last=1, version=1))
    accept_section(stats, 0, pat_section({8: 0x101}, number=1, last=1, version=2), cc=1)
    assert stats.pat == {}
    accept_section(stats, 0, pat_section({9: 0x102}, number=0, last=1, version=2), cc=2)
    assert stats.pat == {8: 0x101, 9: 0x102}


def test_pmt_version_replaces_streams_and_current_next_ignored():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100}))
    accept_section(stats, 0x100, pmt_section({0x200: 0x1B, 0x201: 0x0F}, version=31))
    upcoming = pmt_section({0x202: 0x0F}, current=False)
    accept_section(stats, 0x100, upcoming, cc=1)
    assert pmt_streams(section_packets(0x100, upcoming)[0], 0x100) == {}
    assert stats.elementary_streams == {0x200: 0x1B, 0x201: 0x0F}
    accept_section(stats, 0x100, pmt_section({0x202: 0x0F}), cc=2)
    assert stats.elementary_streams == {0x202: 0x0F}
    accept_section(stats, 0x100, pmt_section({}, version=1), cc=3)
    assert stats.elementary_streams == {}


def test_next_pat_and_network_pid_do_not_advertise_programs():
    stats = TransportStats()
    upcoming = pat_section({7: 0x100}, current=False)
    accept_section(stats, 0, upcoming)
    assert pat_programs(section_packets(0, upcoming)[0]) == {}
    assert stats.pat == {}
    accept_section(stats, 0, pat_section({0: 0x10, 7: 0x100}), cc=1)
    assert stats.pat == {7: 0x100}


def test_pat_removal_and_remapping_discard_stale_pmts():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100, 8: 0x101}))
    accept_section(stats, 0x100, pmt_section({0x200: 0x1B}))
    accept_section(stats, 0x101, pmt_section({0x201: 0x0F}, program=8))
    accept_section(stats, 0, pat_section({7: 0x102}, version=1), cc=1)
    assert stats.pat == {7: 0x102}
    assert stats.elementary_streams == {}
    accept_section(stats, 0x100, pmt_section({0x200: 0x1B}, version=1), cc=1)
    assert stats.elementary_streams == {}
    accept_section(stats, 0x102, pmt_section({0x202: 0x1B}))
    assert stats.elementary_streams == {0x202: 0x1B}
    accept_section(stats, 0, pat_section({}, version=2), cc=2)
    assert stats.pat == {}
    assert stats.elementary_streams == {}
    assert set(stats._assemblers) == {0}


def test_transport_stream_change_discards_pmts_even_when_pat_mapping_matches():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100}, tsid=1))
    accept_section(stats, 0x100, pmt_section({0x200: 0x1B}))
    accept_section(stats, 0, pat_section({7: 0x100}, tsid=2), cc=1)
    assert stats.elementary_streams == {}
    accept_section(stats, 0x100, pmt_section({0x201: 0x0F}), cc=1)
    assert stats.elementary_streams == {0x201: 0x0F}


def test_multiple_programs_share_pmt_pid_but_unadvertised_program_ignored():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100, 8: 0x100}))
    payload = b"\x00" + pmt_section({0x200: 0x1B}) + pmt_section({0x201: 0x0F}, program=8)
    stats.accept(packet(0x100, 0, payload, True))
    assert stats.elementary_streams == {0x200: 0x1B, 0x201: 0x0F}
    accept_section(stats, 0x100, pmt_section({0x202: 0x0F}, program=9), cc=1)
    assert stats.elementary_streams == {0x200: 0x1B, 0x201: 0x0F}
    accept_section(stats, 0x100, pmt_section({0x203: 0x0F}, program=8, version=1), cc=2)
    assert stats.elementary_streams == {0x200: 0x1B, 0x203: 0x0F}


def test_crc_corruption_does_not_publish_and_later_section_recovers():
    stats = TransportStats()
    valid = pat_section({7: 0x100})
    bad = valid[:-1] + bytes((valid[-1] ^ 1,))
    stats.accept(packet(0, 0, b"\x00" + bad, True))
    assert stats.pat == {}
    # Known section boundaries still permit discovery after a bad CRC.
    stats.accept(packet(0, 1, b"\x00" + bad + valid, True))
    assert stats.pat == {7: 0x100}


@pytest.mark.parametrize("interruption", ["gap", "changed_duplicate", "discontinuity", "scrambled", "pointer"])
def test_partial_section_discarded_on_interruption_and_recovers(interruption):
    stats = TransportStats()
    section = pat_section({7: 0x100})
    first = payload_packet(0, 0, b"\x00" + section[:10], start=True)
    stats.accept(first)
    if interruption == "gap":
        stats.accept(packet(0, 2, section[10:]))
        assert stats.continuity_errors == 1
    elif interruption == "changed_duplicate":
        stats.accept(packet(0, 0, section[10:]))
        assert stats.continuity_errors == 1
    elif interruption == "discontinuity":
        stats.accept(payload_packet(0, 7, section[10:], discontinuity=True))
        assert stats.continuity_errors == 0
    elif interruption == "scrambled":
        following = packet(0, 1, section[10:])
        stats.accept(following[:3] + bytes((following[3] | 0x80,)) + following[4:])
    else:
        stats.accept(payload_packet(0, 1, b"\x10\x00", start=True))
        stats.accept(packet(0, 2, section[10:]))
    assert stats.pat == {}
    stats.accept(payload_packet(0, 12, b"\x00" + section, start=True, discontinuity=True))
    assert stats.pat == {7: 0x100}


def test_duplicate_partial_packet_is_not_appended_twice():
    stats = TransportStats()
    section = pat_section({7: 0x100})
    first = payload_packet(0, 0, b"\x00" + section[:6], start=True)
    middle = payload_packet(0, 1, section[6:10])
    for raw in (first, first, middle, middle, packet(0, 2, section[10:])):
        stats.accept(raw)
    assert stats.pat == {7: 0x100}
    assert stats.continuity_errors == 0
    assert stats.packets == 5


def test_continuity_wrap_adaptation_only_discontinuity_and_null_pid():
    stats = TransportStats()
    for raw in (
        packet(0x200, 15),
        payload_packet(0x200, 15),
        payload_packet(0x200, 15),
        packet(0x200, 0),
        payload_packet(0x200, 7, discontinuity=True),
        packet(0x200, 8),
        packet(0x1FFF, 12),
        packet(0x1FFF, 5),
    ):
        stats.accept(raw)
    assert stats.continuity_errors == 0
    stats.accept(payload_packet(0x200, 9))  # adaptation-only must not increment
    assert stats.continuity_errors == 1
    stats.accept(packet(0x200, 10))
    assert stats.continuity_errors == 1


def test_exact_duplicate_allows_changed_pcr_but_not_changed_payload():
    stats = TransportStats()
    first = payload_packet(0x200, 0, b"a", pcr=b"\x00\x00\x00\x00\x7e\x00")
    duplicate = payload_packet(0x200, 0, b"a", pcr=b"\x00\x00\x00\x00\x7e\x01")
    stats.accept(first)
    stats.accept(duplicate)
    assert stats.continuity_errors == 0
    stats.accept(payload_packet(0x200, 0, b"b", pcr=b"\x00\x00\x00\x00\x7e\x02"))
    assert stats.continuity_errors == 1


def test_three_identical_payload_packets_are_not_a_legal_duplicate_pair():
    stats = TransportStats()
    raw = packet(0x200, 0)
    stats.accept(raw)
    stats.accept(raw)
    assert stats.continuity_errors == 0
    stats.accept(raw)
    assert stats.continuity_errors == 1
    stats.accept(packet(0x200, 1))
    assert stats.continuity_errors == 1


def test_external_gap_retains_complete_tables_and_clears_partial_state():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100}))
    accept_section(stats, 0x100, pmt_section({0x200: 0x1B}))
    accept_section(stats, 0, pat_section({8: 0x101}, version=1, number=0, last=1), cc=1)
    replacement = pmt_section({0x201: 0x0F}, version=1)
    stats.accept(payload_packet(0x100, 1, b"\x00" + replacement[:10], start=True))
    stats.reset_partial_sections()
    stats.accept(packet(0x100, 14, replacement[10:]))
    accept_section(stats, 0, pat_section({9: 0x102}, version=1, number=1, last=1), cc=10)
    assert stats.pat == {7: 0x100}
    assert stats.elementary_streams == {0x200: 0x1B}
    assert stats.continuity_errors == 0
    accept_section(stats, 0x100, replacement, cc=15)
    assert stats.elementary_streams == {0x201: 0x0F}


def test_psi_length_is_bounded_and_recovers_at_next_start():
    stats = TransportStats()
    stats.accept(packet(0, 0, b"\x00\x00\xbf\xff", True))
    for cc in range(1, 10):
        stats.accept(packet(0, cc, b"\x00" * 184))
    assert stats.pat == {}
    assert len(stats._assemblers[0].buffer) == 0
    accept_section(stats, 0, pat_section({7: 0x100}), cc=10)
    assert stats.pat == {7: 0x100}


def test_maximum_sized_pat_section_is_supported():
    programs = {n: 0x100 + n for n in range(1, 254)}
    section = pat_section(programs)
    assert len(section) == 1024
    stats = TransportStats()
    for raw in section_packets(0, section):
        stats.accept(raw)
        assert len(stats._assemblers[0].buffer) <= 1024
    assert stats.pat == programs


@pytest.mark.parametrize("contents", [
    "0001c100000007e10000",  # incomplete program entry
    "0001c1000000070100",  # missing reserved PID bits
    "0001c102010007e100",  # section number > last section
    "0001c100000007ffff",  # null PMT PID
])
def test_structurally_invalid_pat_with_valid_crc_is_ignored(contents):
    raw = section_packet(0, 0, bytes.fromhex(contents))
    assert pat_programs(raw) == {}
    stats = TransportStats()
    stats.accept(raw)
    assert stats.pat == {}


@pytest.mark.parametrize("contents", [
    "0007c10000e200f001ff",  # truncated program descriptor
    "0007c10000e200f0020102",  # descriptor payload exceeds loop
    "0007c10000e200f0001be200f0020101",  # truncated ES descriptor
    "0007c10000e200f0001be200f001",  # truncated ES info
    "0007c10001e200f000",  # PMT must have a single section
    "0007c10000e200f0001b",  # incomplete ES entry
])
def test_structurally_invalid_pmt_with_valid_crc_is_ignored(contents):
    raw = section_packet(2, 0x100, bytes.fromhex(contents))
    assert pmt_streams(raw, 0x100) == {}
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100}))
    stats.accept(raw)
    assert stats.elementary_streams == {}


def test_pmt_program_descriptors_can_cross_packet_boundary():
    # Unknown descriptors are skipped by their lengths, without decoding values.
    descriptors = b"\x80\xfa" + b"d" * 250
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100}))
    accept_section(stats, 0x100, pmt_section({0x200: 0x1B}, descriptors=descriptors))
    assert stats.elementary_streams == {0x200: 0x1B}


@pytest.mark.parametrize("control,length,flags", [
    (0x20, 0, 0),  # adaptation-only must occupy all remaining bytes
    (0x20, 182, 0),
    (0x30, 183, 0),  # payload control must leave payload bytes
    (0x30, 184, 0),
    (0x30, 1, 0x10),  # PCR is flagged but truncated
    (0x30, 1, 0x08),  # OPCR is flagged but truncated
    (0x30, 1, 0x04),  # splice countdown is flagged but truncated
    (0x30, 1, 0x02),  # no private-data length
    (0x30, 1, 0x01),  # no adaptation-extension length
])
def test_adaptation_bounds_and_optional_fields_are_validated(control, length, flags):
    raw = bytes((0x47, 0x01, 0x00, control, length, flags)) + b"\xff" * 182
    with pytest.raises(ValueError):
        parse_packet(raw)


def test_valid_empty_adaptation_field_and_discontinuity_flag():
    raw = bytes((0x47, 0x01, 0x00, 0x30, 0)) + b"\xff" * 183
    parsed = parse_packet(raw)
    assert len(parsed.payload) == 183
    assert not parsed.discontinuity_indicator
    assert parse_packet(payload_packet(0x100, 0, discontinuity=True)).discontinuity_indicator


def test_orphan_continuations_and_sections_without_start_are_ignored():
    stats = TransportStats()
    section = pat_section({7: 0x100})
    stats.accept(packet(0, 0, section))
    assert stats.pat == {}
    first = pat_section({7: 0x100}, number=0, last=1)
    second = pat_section({8: 0x101}, number=1, last=1)
    stats.accept(payload_packet(0, 1, b"\x00" + first[:10], start=True))
    stats.accept(packet(0, 2, first[10:] + second))
    # A new section in the continuation without PUSI is not a valid boundary.
    assert stats.pat == {}
    accept_section(stats, 0, second, cc=3)
    assert stats.pat == {7: 0x100, 8: 0x101}


def test_duplicate_discontinuity_packet_does_not_reset_partial_section_twice():
    stats = TransportStats()
    section = pat_section({7: 0x100})
    first = payload_packet(0, 5, b"\x00" + section[:10], start=True, discontinuity=True)
    stats.accept(first)
    stats.accept(first)
    stats.accept(packet(0, 6, section[10:]))
    assert stats.pat == {7: 0x100}
    assert stats.continuity_errors == 0


def test_adaptation_only_packet_preserves_pending_psi():
    stats = TransportStats()
    section = pat_section({7: 0x100})
    stats.accept(payload_packet(0, 3, b"\x00" + section[:10], start=True))
    stats.accept(payload_packet(0, 3))
    stats.accept(packet(0, 4, section[10:]))
    assert stats.pat == {7: 0x100}
    assert stats.continuity_errors == 0


def test_new_start_does_not_join_an_unfinished_previous_section():
    stats = TransportStats()
    previous = pat_section({7: 0x100})
    stats.accept(payload_packet(0, 0, b"\x00" + previous[:10], start=True))
    accept_section(stats, 0, pat_section({8: 0x101}), cc=1)
    assert stats.pat == {8: 0x101}


def test_bad_table_syntax_and_scrambled_psi_are_ignored():
    section = pat_section({7: 0x100})
    bad = bytes((section[0], section[1] ^ 0x80)) + section[2:-4]
    bad += mpeg_crc32(bad).to_bytes(4, "big")
    stats = TransportStats()
    stats.accept(packet(0, 0, b"\x00" + bad, True))
    assert stats.pat == {}
    raw = packet(0, 1, b"\x00" + section, True)
    scrambled = raw[:3] + bytes((raw[3] | 0x80,)) + raw[4:]
    assert pat_programs(scrambled) == {}
    stats.accept(scrambled)
    assert stats.pat == {}
    accept_section(stats, 0, section, cc=2)
    assert stats.pat == {7: 0x100}


def test_conflicting_programs_across_pat_sections_do_not_publish():
    stats = TransportStats()
    accept_section(stats, 0, pat_section({7: 0x100}, number=0, last=1))
    accept_section(stats, 0, pat_section({7: 0x101}, number=1, last=1), cc=1)
    assert stats.pat == {}
    accept_section(stats, 0, pat_section({7: 0x100}, version=1), cc=2)
    assert stats.pat == {7: 0x100}

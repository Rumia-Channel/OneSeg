"""MPEG-2 TS framing, validation, PAT/PMT discovery and file output.

These functions consume *already demodulated* 188-byte packets. They do not
convert RF I/Q into MPEG-TS or decode video/audio elementary streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKET_BYTES = 188
SYNC = 0x47


def mpeg_crc32(data: bytes) -> int:
    """CRC-32/MPEG-2: poly 0x04C11DB7, init FFFFFFFF, no reflect/xorout."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)) & 0xFFFFFFFF
    return crc


@dataclass(frozen=True)
class Packet:
    raw: bytes
    pid: int
    payload_unit_start: bool
    continuity_counter: int
    payload: bytes
    scrambling: int
    discontinuity_indicator: bool = False


def parse_packet(raw: bytes) -> Packet:
    if len(raw) != PACKET_BYTES or raw[0] != SYNC:
        raise ValueError("not an aligned 188-byte MPEG-TS packet")
    if raw[1] & 0x80:
        raise ValueError("transport-error indicator set")
    pid = (raw[1] & 0x1F) << 8 | raw[2]
    adaptation = (raw[3] >> 4) & 3
    if adaptation == 0:
        raise ValueError("reserved adaptation-field control")
    pos = 4
    discontinuity = False
    if adaptation & 2:
        length = raw[pos]
        if (adaptation == 2 and length != 183) or (adaptation == 3 and length > 182):
            raise ValueError("invalid adaptation length")
        if length:
            flags = raw[pos + 1]
            discontinuity = bool(flags & 0x80)
            required = 1 + 6 * bool(flags & 0x10) + 6 * bool(flags & 0x08) + bool(flags & 0x04)
            if required > length:
                raise ValueError("truncated adaptation field")
            at = pos + 1 + required
            for flag in (0x02, 0x01):  # private data and adaptation extension
                if flags & flag:
                    if at >= pos + 1 + length:
                        raise ValueError("truncated adaptation field")
                    at += 1 + raw[at]
            if at > pos + 1 + length:
                raise ValueError("truncated adaptation field")
        pos += length + 1
        if pos > PACKET_BYTES:
            raise ValueError("invalid adaptation length")
    payload = raw[pos:] if adaptation & 1 else b""
    return Packet(
        raw=raw,
        pid=pid,
        payload_unit_start=bool(raw[1] & 0x40),
        continuity_counter=raw[3] & 0x0F,
        payload=payload,
        scrambling=(raw[3] >> 6) & 3,
        discontinuity_indicator=discontinuity,
    )


class TransportFramer:
    """Incremental 188-byte TS parser; tolerate partial read/packet misalignment."""

    def __init__(self):
        self.buffer = bytearray()
        self.packets = 0
        self.bad_packets = 0
        self.dropped_bytes = 0

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer.extend(data)
        output = []
        while len(self.buffer) >= PACKET_BYTES:
            if self.buffer[0] != SYNC or (
                len(self.buffer) >= 2 * PACKET_BYTES
                and self.buffer[PACKET_BYTES] != SYNC
            ):
                del self.buffer[0]
                self.dropped_bytes += 1
                continue
            raw = bytes(self.buffer[:PACKET_BYTES])
            try:
                parse_packet(raw)
            except ValueError:
                del self.buffer[0]
                self.bad_packets += 1
                continue
            del self.buffer[:PACKET_BYTES]
            output.append(raw)
            self.packets += 1
        return output


# PAT and PMT section_length is limited to 1021 by ISO/IEC 13818-1.
_MAX_PSI_SECTION_LENGTH = 1021


class _SectionAssembler:
    """One bounded, incomplete PSI section per PID; never guess a boundary."""

    def __init__(self):
        self.buffer = bytearray()

    def reset(self) -> None:
        self.buffer.clear()

    def _consume(self, data: bytes, *, new_sections: bool) -> list[bytes]:
        sections = []
        at = 0
        while at < len(data):
            if not self.buffer and (not new_sections or data[at] == 0xFF):
                break
            if len(self.buffer) < 3:
                count = min(3 - len(self.buffer), len(data) - at)
                self.buffer.extend(data[at:at + count])
                at += count
                if len(self.buffer) < 3:
                    break
            size = ((self.buffer[1] & 0x0F) << 8) | self.buffer[2]
            if not 4 <= size <= _MAX_PSI_SECTION_LENGTH:
                # A corrupt length provides no trustworthy following boundary.
                self.reset()
                break
            count = min(3 + size - len(self.buffer), len(data) - at)
            self.buffer.extend(data[at:at + count])
            at += count
            if len(self.buffer) != 3 + size:
                break
            section = bytes(self.buffer)
            self.reset()
            if mpeg_crc32(section) == 0:
                sections.append(section)
            if not new_sections:
                break
        return sections

    def feed(self, packet: Packet) -> list[bytes]:
        if packet.scrambling:
            self.reset()
            return []
        if not packet.payload:
            return []
        if not packet.payload_unit_start:
            return self._consume(packet.payload, new_sections=False)
        pointer = packet.payload[0]
        start = 1 + pointer
        if start >= len(packet.payload):
            self.reset()
            return []
        # Bytes before the pointer may finish the preceding section. They must
        # never be interpreted as the beginning of another section.
        sections = self._consume(packet.payload[1:start], new_sections=False)
        self.reset()
        sections.extend(self._consume(packet.payload[start:], new_sections=True))
        return sections


def _psi_section(packet: Packet) -> bytes | None:
    """Return the first complete, CRC-checked section in this single packet."""
    sections = _SectionAssembler().feed(packet)
    return sections[0] if sections else None


def _valid_table(section: bytes, table_id: int, minimum: int) -> bool:
    return (
        len(section) >= minimum
        and section[0] == table_id
        and section[1] & 0xF0 == 0xB0
        and section[5] & 0xC1 == 0xC1  # reserved bits and current_next_indicator
        and section[6] <= section[7]
    )


def _pat_entries(section: bytes) -> dict[int, int] | None:
    if not _valid_table(section, 0, 12) or (len(section) - 12) % 4:
        return None
    programs = {}
    for at in range(8, len(section) - 4, 4):
        if section[at + 2] & 0xE0 != 0xE0:
            return None
        program = (section[at] << 8) | section[at + 1]
        pid = ((section[at + 2] & 0x1F) << 8) | section[at + 3]
        if program:
            if pid == 0 or pid == 0x1FFF:
                return None
            if program in programs and programs[program] != pid:
                return None
            programs[program] = pid
    return programs


def _valid_descriptors(section: bytes, start: int, end: int) -> bool:
    if end > len(section) - 4:
        return False
    while start < end:
        if start + 2 > end:
            return False
        start += 2 + section[start + 1]
    return start == end


def _pmt_entries(section: bytes) -> dict[int, int] | None:
    if (
        not _valid_table(section, 2, 16)
        or section[3:5] == b"\x00\x00"
        or section[6] != 0 or section[7] != 0
        or section[8] & 0xE0 != 0xE0
        or section[10] & 0xF0 != 0xF0
    ):
        return None
    program_info_length = ((section[10] & 0x0F) << 8) | section[11]
    at = 12 + program_info_length
    end = len(section) - 4
    if not _valid_descriptors(section, 12, at):
        return None
    streams = {}
    while at + 5 <= end:
        if section[at + 1] & 0xE0 != 0xE0 or section[at + 3] & 0xF0 != 0xF0:
            return None
        stream_type = section[at]
        pid = ((section[at + 1] & 0x1F) << 8) | section[at + 2]
        length = ((section[at + 3] & 0x0F) << 8) | section[at + 4]
        following = at + 5 + length
        if not _valid_descriptors(section, at + 5, following):
            return None
        if pid == 0 or pid == 0x1FFF or (pid in streams and streams[pid] != stream_type):
            return None
        streams[pid] = stream_type
        at = following
    return streams if at == end else None


def pat_programs(raw: bytes) -> dict[int, int]:
    """PAT -> {program_number: PMT_PID} from complete sections in one packet.

    Use TransportStats for incremental multi-packet and multi-section discovery.
    """
    packet = parse_packet(raw)
    if packet.pid != 0:
        return {}
    result = {}
    for section in _SectionAssembler().feed(packet):
        entries = _pat_entries(section)
        if entries is not None:
            result.update(entries)
    return result


def pmt_streams(raw: bytes, pmt_pid: int) -> dict[int, int]:
    """PMT -> {elementary_PID: stream_type} from complete sections in one packet."""
    packet = parse_packet(raw)
    if packet.pid != pmt_pid:
        return {}
    streams = {}
    for section in _SectionAssembler().feed(packet):
        entries = _pmt_entries(section)
        if entries is not None:
            streams.update(entries)
    return streams


@dataclass
class _PATCollector:
    """Publish only complete current versions; keep at most one pending table."""

    active: tuple[int, int, int] | None = None
    pending: tuple[int, int, int] | None = None
    sections: dict[int, dict[int, int]] = field(default_factory=dict)

    def accept(self, section: bytes) -> dict[int, int] | None:
        entries = _pat_entries(section)
        if entries is None:
            return None
        key = ((section[3] << 8) | section[4], (section[5] >> 1) & 31, section[7])
        if self.active is not None and key[:2] == self.active[:2]:
            return None
        if key != self.pending:
            self.pending = key
            self.sections.clear()
        number = section[6]
        if number in self.sections and self.sections[number] != entries:
            # The same section/version cannot describe conflicting programs.
            self.sections.clear()
        self.sections[number] = entries
        if len(self.sections) != key[2] + 1:
            return None
        programs = {}
        for number in range(key[2] + 1):
            for program, pid in self.sections[number].items():
                if program in programs and programs[program] != pid:
                    self.sections.clear()
                    return None
                programs[program] = pid
        self.active = key
        self.pending = None
        self.sections.clear()
        return programs


def _duplicate_key(packet: Packet) -> bytes:
    # ISO/IEC 13818-1 permits an updated PCR in an otherwise identical duplicate.
    raw = packet.raw
    if raw[3] & 0x20 and raw[4] >= 7 and raw[5] & 0x10:
        return raw[:6] + b"\x00" * 6 + raw[12:]
    return raw


@dataclass
class TransportStats:
    """Packet statistics and CRC-checked, current PAT/PMT service discovery.

    Partial sections are bounded to 1024 bytes per advertised PSI PID. PAT
    versions are installed atomically after all sections arrive; PMT versions
    replace each program's streams instead of accumulating stale entries.
    """

    packets: int = 0
    continuity_errors: int = 0
    pat: dict[int, int] = field(default_factory=dict)
    elementary_streams: dict[int, int] = field(default_factory=dict)
    _previous: dict[int, int] = field(default_factory=dict, repr=False)
    _previous_packets: dict[int, bytes] = field(default_factory=dict, repr=False)
    _duplicated: set[int] = field(default_factory=set, repr=False)
    _assemblers: dict[int, _SectionAssembler] = field(default_factory=dict, repr=False)
    _pat_table: _PATCollector = field(default_factory=_PATCollector, repr=False)
    _pmt_versions: dict[int, int] = field(default_factory=dict, repr=False)
    _program_streams: dict[int, dict[int, int]] = field(default_factory=dict, repr=False)

    def reset_partial_sections(self) -> None:
        """Mark an external input gap, retaining fully validated service tables.

        Independently decoded capture windows must not join PSI fragments or
        compare continuity counters across their unverified boundaries.
        """
        for assembler in self._assemblers.values():
            assembler.reset()
        self._pat_table.pending = None
        self._pat_table.sections.clear()
        self._previous.clear()
        self._previous_packets.clear()
        self._duplicated.clear()

    def _reset_section(self, pid: int) -> None:
        assembler = self._assemblers.get(pid)
        if assembler is not None:
            assembler.reset()

    def _check_continuity(self, packet: Packet) -> bool:
        """Return true for a duplicate, which must not be fed into PSI twice."""
        pid = packet.pid
        if pid == 0x1FFF:
            return False
        previous = self._previous.get(pid)
        key = _duplicate_key(packet)
        if packet.payload and previous == packet.continuity_counter and (
            self._previous_packets.get(pid) == key
        ):
            # Only one duplicate is allowed for a given original TS packet.
            if pid in self._duplicated:
                self.continuity_errors += 1
                self._reset_section(pid)
            self._duplicated.add(pid)
            return True
        self._duplicated.discard(pid)
        if packet.discontinuity_indicator:
            self._reset_section(pid)
        elif previous is not None and (
            (previous + bool(packet.payload)) & 15
        ) != packet.continuity_counter:
            self.continuity_errors += 1
            self._reset_section(pid)
        self._previous[pid] = packet.continuity_counter
        self._previous_packets[pid] = key
        return False

    def _refresh_streams(self) -> None:
        self.elementary_streams.clear()
        for streams in self._program_streams.values():
            self.elementary_streams.update(streams)

    def _accept_pat(self, section: bytes) -> None:
        old_table = self._pat_table.active
        programs = self._pat_table.accept(section)
        if programs is None:
            return
        same_transport = old_table is not None and old_table[0] == self._pat_table.active[0]
        unchanged = {
            program for program, pid in programs.items()
            if same_transport and self.pat.get(program) == pid
        }
        self._program_streams = {
            program: streams for program, streams in self._program_streams.items()
            if program in unchanged
        }
        self._pmt_versions = {
            program: version for program, version in self._pmt_versions.items()
            if program in unchanged
        }
        # Retain an incomplete PMT only when its PID still serves the same
        # programs in the same transport; a remapped PID may carry old bytes.
        retained_pids = {programs[program] for program in unchanged}
        self._assemblers = {
            pid: assembler for pid, assembler in self._assemblers.items()
            if pid == 0 or pid in retained_pids
        }
        self.pat.clear()
        self.pat.update(programs)
        self._refresh_streams()

    def accept(self, raw: bytes) -> None:
        packet = parse_packet(raw)
        self.packets += 1
        if self._check_continuity(packet):
            return
        if packet.pid != 0 and packet.pid not in self.pat.values():
            return
        assembler = self._assemblers.setdefault(packet.pid, _SectionAssembler())
        for section in assembler.feed(packet):
            if packet.pid == 0:
                self._accept_pat(section)
                continue
            streams = _pmt_entries(section)
            if streams is None:
                continue
            program = (section[3] << 8) | section[4]
            if self.pat.get(program) != packet.pid:
                continue
            version = (section[5] >> 1) & 31
            if self._pmt_versions.get(program) == version:
                continue
            self._pmt_versions[program] = version
            self._program_streams[program] = streams
            self._refresh_streams()

class TransportWriter:
    """Write only validated 188-byte TS packets. Never synthesize MPEG video."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        self.stats = TransportStats()

    def write(self, raw: bytes) -> None:
        self.stats.accept(raw)
        self._file.write(raw)

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

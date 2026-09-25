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
    if adaptation & 2:
        if pos >= PACKET_BYTES:
            raise ValueError("truncated adaptation field")
        length = raw[pos]
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


def _psi_section(packet: Packet) -> bytes | None:
    """Return a complete PSI section when contained within a single packet.

    Multi-packet PSI sections require an assembler; do not return truncated PSI.
    """
    if not packet.payload_unit_start or not packet.payload:
        return None
    pointer = packet.payload[0]
    begin = 1 + pointer
    if begin + 3 > len(packet.payload):
        return None
    size = (packet.payload[begin + 1] & 0x0F) << 8 | packet.payload[begin + 2]
    end = begin + 3 + size
    if end > len(packet.payload) or size < 4:
        return None
    section = packet.payload[begin:end]
    if mpeg_crc32(section) != 0:
        return None
    return section


def pat_programs(raw: bytes) -> dict[int, int]:
    """PAT (PID 0) -> {program_number: PMT_PID}; empty if no valid single-packet PAT."""
    packet = parse_packet(raw)
    if packet.pid != 0:
        return {}
    section = _psi_section(packet)
    if section is None or section[0] != 0 or len(section) < 12:
        return {}
    result = {}
    for at in range(8, len(section) - 4, 4):
        if at + 4 > len(section) - 4:
            break
        program = (section[at] << 8) | section[at + 1]
        pid = ((section[at + 2] & 0x1F) << 8) | section[at + 3]
        if program:
            result[program] = pid
    return result


def pmt_streams(raw: bytes, pmt_pid: int) -> dict[int, int]:
    """PMT -> {elementary_PID: stream_type}; single-packet PMT only."""
    packet = parse_packet(raw)
    if packet.pid != pmt_pid:
        return {}
    section = _psi_section(packet)
    if section is None or section[0] != 2 or len(section) < 16:
        return {}
    program_info_length = (section[10] & 0x0F) << 8 | section[11]
    at = 12 + program_info_length
    end = len(section) - 4
    streams = {}
    while at + 5 <= end:
        stream_type = section[at]
        pid = ((section[at + 1] & 0x1F) << 8) | section[at + 2]
        length = ((section[at + 3] & 0x0F) << 8) | section[at + 4]
        if at + 5 + length > end:
            return {}
        streams[pid] = stream_type
        at += 5 + length
    return streams if at == end else {}


@dataclass
class TransportStats:
    packets: int = 0
    continuity_errors: int = 0
    pat: dict[int, int] = field(default_factory=dict)
    elementary_streams: dict[int, int] = field(default_factory=dict)
    _previous: dict[int, int] = field(default_factory=dict, repr=False)

    def accept(self, raw: bytes) -> None:
        packet = parse_packet(raw)
        self.packets += 1
        if packet.payload and packet.pid != 0x1FFF:
            old = self._previous.get(packet.pid)
            if old is not None and (old + 1) % 16 != packet.continuity_counter:
                self.continuity_errors += 1
            self._previous[packet.pid] = packet.continuity_counter
        if packet.pid == 0:
            self.pat.update(pat_programs(raw))
        for pid in set(self.pat.values()):
            if packet.pid == pid:
                self.elementary_streams.update(pmt_streams(raw, pid))


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

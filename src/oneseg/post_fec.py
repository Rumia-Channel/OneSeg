"""Last ISDB-T receiver stages when the inner FEC *already produced bytes*.

An actual RTL-SDR IQ -> TS receiver still needs validated TMCC, carrier
equalization, bit/time/frequency interleaver reversal, depuncturing and
byte/frame alignment upstream of this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .fec import OuterReedSolomon
from .ts import parse_packet


class ByteDeinterleaver:
    """12-way, 17-byte-increment convolutional byte deinterleaver."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.index = 0
        self.buffers = [deque([0] * (17 * (11 - branch)))
                        for branch in range(12)]

    def feed(self, data: bytes) -> bytes:
        output = bytearray(len(data))
        for i, b in enumerate(data):
            branch = self.index % 12
            fifo = self.buffers[branch]
            fifo.append(b)
            output[i] = fifo.popleft()
            self.index += 1
        return bytes(output)


class EnergyDescrambler:
    """ISDB-T packet energy descrambler with periodic frame-boundary reset.

    Input is 204-byte blocks after byte deinterleaving, with the prior sync
    byte at position 203. The 203 other bytes are XORed with the continuing
    x^15+x^14+1 PRBS. The skipped sync position still advances the PRBS.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.register = 0xA9

    def _clock(self) -> int:
        byte = 0
        for _ in range(8):
            feedback = ((self.register >> 13) ^ (self.register >> 14)) & 1
            self.register = ((self.register << 1) | feedback) & 0x7FFF
            byte = (byte << 1) | feedback
        return byte

    def packet(self, incoming: bytes) -> bytes:
        if len(incoming) != 204:
            raise ValueError("energy descrambler requires a 204-byte input block")
        transformed = bytes([incoming[-1]]) + bytes(
            incoming[i] ^ self._clock() for i in range(203)
        )
        self._clock()
        return transformed


@dataclass
class PostFecStats:
    good_ts_packets: int = 0
    rejected_rs_words: int = 0
    rejected_ts_packets: int = 0


class PostViterbiTransport:
    """Streaming byte-deinterleaver -> descrambler -> RS -> 188-byte MPEG-TS.

    The caller MUST provide correctly aligned Viterbi-output bytes and
    explicit ISDB-T frame boundary (start=True) when known. This class cannot
    infer the OFDM frame boundary from a random midstream bitstream.
    """

    def __init__(self, *, byte_deinterleaving: bool = True):
        self.byte_deinterleaving = byte_deinterleaving
        self.byte_deinterleaver = ByteDeinterleaver()
        self.energy = EnergyDescrambler()
        self.rs = OuterReedSolomon()
        self.pending = bytearray()
        self.stats = PostFecStats()

    def reset(self):
        self.byte_deinterleaver.reset()
        self.energy.reset()
        self.pending.clear()
        self.stats = PostFecStats()

    def feed(self, data: bytes, *, frame_begin: bool = False) -> list[bytes]:
        if frame_begin:
            if self.pending:
                raise ValueError("new ISDB-T frame starts at an unaligned 204-byte boundary")
            self.energy.reset()
        if self.byte_deinterleaving:
            data = self.byte_deinterleaver.feed(data)
        self.pending.extend(data)
        output = []
        while len(self.pending) >= 204:
            block = bytes(self.pending[:204])
            del self.pending[:204]
            unmasked = self.energy.packet(block)
            try:
                corrected = self.rs.decode(unmasked)
            except ValueError:
                self.stats.rejected_rs_words += 1
                continue
            try:
                parse_packet(corrected)
            except ValueError:
                self.stats.rejected_ts_packets += 1
                continue
            self.stats.good_ts_packets += 1
            output.append(corrected)
        return output

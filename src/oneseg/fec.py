"""Independent ISDB-T outer Reed–Solomon blocks and PRBS experiments.

An RS(204,188) codeword is not an MPEG-TS packet until RF carrier recovery,
TMCC, deinterleaving, inner Viterbi FEC and energy descrambling have succeeded.
"""

from __future__ import annotations

from reedsolo import RSCodec, ReedSolomonError

from .ts import PACKET_BYTES, SYNC

SHORTENING = 51
OUTER_CODEWORD_BYTES = 204
OUTER_PARITY_BYTES = 16


class OuterReedSolomon:
    """GF(256), primitive 0x11d, fcr=0 shortened RS(255,239) -> (204,188)."""

    def __init__(self):
        self._codec = RSCodec(
            nsym=OUTER_PARITY_BYTES, nsize=255, fcr=0, prim=0x11D
        )

    def encode_test_vector(self, source_188: bytes) -> bytes:
        """Construct synthetic codewords for FEC tests; NOT an ISDB-T RF modulator."""
        if len(source_188) != PACKET_BYTES:
            raise ValueError("expected 188 data bytes")
        full = bytes(SHORTENING) + source_188
        return bytes(self._codec.encode(full)[SHORTENING:])

    def decode(self, codeword: bytes) -> bytes:
        """Correct 0-8 unknown byte errors; fail closed on uncorrectable codewords."""
        if len(codeword) != OUTER_CODEWORD_BYTES:
            raise ValueError("expected 204 RS-coded bytes")
        try:
            message, _, _ = self._codec.decode(bytes(SHORTENING) + codeword)
        except ReedSolomonError as exc:
            raise ValueError("uncorrectable RS codeword") from exc
        if bytes(message[:SHORTENING]) != bytes(SHORTENING):
            raise ValueError("RS correction changed the shortened zero prefix")
        return bytes(message[SHORTENING:SHORTENING + PACKET_BYTES])


def prbs_bytes(count: int, register: int = 0xA9) -> bytes:
    """PRBS polynomial x^15+x^14+1 used by ISDB-T energy dispersal.

    The register is reset at an ISDB-T frame boundary; callers must not reset
    it per arbitrary chunk and must handle the MPEG sync-byte convention.
    """
    if not 0 <= count or not 0 <= register <= 0x7FFF:
        raise ValueError("invalid PRBS request")
    out = bytearray(count)
    for i in range(count):
        byte = 0
        for _ in range(8):
            feedback = ((register >> 13) ^ (register >> 14)) & 1
            register = ((register << 1) | feedback) & 0x7FFF
            byte = (byte << 1) | feedback
        out[i] = byte
    return bytes(out)

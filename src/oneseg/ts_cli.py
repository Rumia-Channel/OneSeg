"""Inspect recovered MPEG-TS and convert *post-inner-FEC* RS codewords to TS.

This CLI does not decode I/Q: --rs204 requires a 204-byte-per-codeword input
after the ISDB-T RF/OFDM/inner-FEC stages. It is intended to verify downstream
components independently with synthetic and future demodulator test vectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .fec import OuterReedSolomon
from .ts import TransportFramer, TransportStats, TransportWriter, parse_packet


def decode_outer_file(source: Path, target: Path, *, strict: bool = True) -> tuple[int, int]:
    decoder = OuterReedSolomon()
    accepted = rejected = 0
    with source.open("rb") as inp, TransportWriter(target) as out:
        while data := inp.read(204):
            if len(data) != 204:
                if strict:
                    raise ValueError("trailing partial RS codeword")
                rejected += 1
                break
            try:
                packet = decoder.decode(data)
                parse_packet(packet)
            except ValueError:
                rejected += 1
                if strict:
                    raise
                continue
            out.write(packet)
            accepted += 1
    return accepted, rejected


def inspect_ts(source: Path) -> tuple[TransportStats, TransportFramer]:
    stats = TransportStats()
    framer = TransportFramer()
    with source.open("rb") as stream:
        while data := stream.read(256 * 1024):
            for packet in framer.feed(data):
                stats.accept(packet)
    return stats, framer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    convert = subparsers.add_parser(
        "rs204", help="decode pre-existing shortened outer RS(204,188) codewords"
    )
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--skip-bad", action="store_true", help="skip uncorrectable codewords")
    inspect = subparsers.add_parser("inspect", help="inspect an already-demodulated .ts")
    inspect.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        if args.task == "rs204":
            accepted, dropped = decode_outer_file(
                args.input, args.output, strict=not args.skip_bad
            )
            print(f"TS packets: {accepted}; rejected RS words: {dropped}; output: {args.output}")
            if not accepted:
                return 2
        else:
            stats, framer = inspect_ts(args.input)
            print(
                f"TS packets: {stats.packets}; framing losses: {framer.dropped_bytes}; "
                f"continuity errors: {stats.continuity_errors}"
            )
            print(f"PAT: {stats.pat}; elementary streams: {stats.elementary_streams}")
            if not stats.packets:
                return 2
    except (OSError, ValueError) as exc:
        parser.exit(2, f"TS processing failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

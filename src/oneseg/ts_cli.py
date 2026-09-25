"""Inspect recovered MPEG-TS and convert *post-inner-FEC* RS codewords to TS.

This CLI does not decode I/Q: --rs204 requires a 204-byte-per-codeword input
after the ISDB-T RF/OFDM/inner-FEC stages. It is intended to verify downstream
components independently with synthetic and future demodulator test vectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .fec import OuterReedSolomon
from .post_fec import PostViterbiTransport
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


def decode_post_fec_file(
    source: Path, target: Path, *, byte_deinterleaving: bool = True
) -> tuple[int, int]:
    """Consume externally recovered, aligned Viterbi bytes (NOT RTL-SDR I/Q)."""
    decoder = PostViterbiTransport(byte_deinterleaving=byte_deinterleaving)
    with source.open("rb") as inp, TransportWriter(target) as out:
        first = True
        while chunk := inp.read(204 * 64):
            for packet in decoder.feed(chunk, frame_begin=first):
                out.write(packet)
            first = False
    if decoder.pending:
        raise ValueError("trailing partial 204-byte block")
    rejected = decoder.stats.rejected_rs_words + decoder.stats.rejected_ts_packets
    return decoder.stats.good_ts_packets, rejected


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
    postfec = subparsers.add_parser(
        "postfec",
        help="aligned Viterbi-output bytes -> byte/energy deinterleave, RS, TS",
    )
    postfec.add_argument("input", type=Path)
    postfec.add_argument("output", type=Path)
    postfec.add_argument(
        "--no-byte-deinterleave",
        action="store_true",
        help="input is already deinterleaved; still needs descrambling/RS",
    )
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
        elif args.task == "postfec":
            count, rejected = decode_post_fec_file(
                args.input, args.output,
                byte_deinterleaving=not args.no_byte_deinterleave,
            )
            print(f"TS packets: {count}; rejected blocks: {rejected}; output: {args.output}")
            if not count:
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

"""Recover only RS-validated partial one-seg TS from a deinterleaved fixture.

Mode-3 Layer-A QPSK 2/3, with verified TMCC frame-start annotations.
The tuner is NOT accessed; the output is offline and may lack PAT/PMT
or entire PES packets. Bad packets are omitted, NEVER synthesized.

Receiver stages: Viterbi -> 12-way byte deinterleaving -> ISDB-T energy
descrambling -> shortened RS(204,188) -> MPEG-TS packet validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from .fec import OuterReedSolomon, prbs_bytes
from .post_fec import ByteDeinterleaver
from .ts import TransportStats, TransportWriter, parse_packet
from .viterbi import decode_soft_bits

PACKET_SIZE = 204
FRAME_PACKETS = 64
BYTE_DEINTERLEAVER_WARMUP = 11
CALIBRATION_PACKETS = 48
MIN_SYNC = 12


def sync_byte_phase(decoded: np.ndarray) -> tuple[int, int]:
    """Find periodic raw Viterbi-output syncs; not itself a TS proof."""
    raw = np.asarray(decoded, dtype=np.uint8).reshape(-1)
    if len(raw) < 204 * 16:
        raise ValueError("need at least 16 candidate 204-byte packets")
    prefix = raw[:min(len(raw), PACKET_SIZE * FRAME_PACKETS)]
    hits = np.array([
        np.count_nonzero(prefix[offset::PACKET_SIZE] == 0x47)
        for offset in range(PACKET_SIZE)
    ])
    best = int(np.argmax(hits))
    count = int(hits[best])
    opportunities = len(prefix[best::PACKET_SIZE])
    if count < MIN_SYNC or count < opportunities * 0.60:
        raise ValueError(
            f"no reliable 204-byte 0x47 sync alignment "
            f"({count}/{opportunities} at best phase)"
        )
    return best, count


def byte_deinterleaved_blocks(
    decoded: np.ndarray, sync_position: int
) -> np.ndarray:
    """Undo 12-way byte delay, placing 0x47 at the end of each 204 block."""
    raw = np.asarray(decoded, dtype=np.uint8).reshape(-1)
    if not 0 <= sync_position < PACKET_SIZE:
        raise ValueError("bad byte sync phase")
    # A received sync at position P ends its encoded 204-byte block.
    # The following input byte is the first of the next complete block.
    begin = (sync_position + 1) % PACKET_SIZE
    stream = bytes(raw[begin:])
    deinterleaved = ByteDeinterleaver().feed(stream)
    full_blocks = len(deinterleaved) // PACKET_SIZE
    if full_blocks <= BYTE_DEINTERLEAVER_WARMUP + 4:
        raise ValueError("not enough bytes after 12-way byte deinterleaver warmup")
    # The sync byte was delayed by zero in branch 11. The RS/energy
    # data block is [sync-last-byte, first 203 deinterleaved bytes].
    rows = np.frombuffer(
        deinterleaved[:full_blocks * PACKET_SIZE], dtype=np.uint8
    ).reshape(-1, PACKET_SIZE)
    coded = np.empty_like(rows)
    coded[:, 0] = rows[:, -1]
    coded[:, 1:] = rows[:, :-1]
    return coded


def _unmask_blocks(blocks: np.ndarray, phase: int) -> np.ndarray:
    """Undo the 203 scrambled bytes, advancing the PRBS through sync."""
    if not 0 <= phase < FRAME_PACKETS:
        raise ValueError("invalid 64-packet energy-dispersal phase")
    mask = np.frombuffer(
        prbs_bytes(FRAME_PACKETS * PACKET_SIZE), dtype=np.uint8
    ).reshape(FRAME_PACKETS, PACKET_SIZE)
    result = np.array(blocks, dtype=np.uint8, copy=True)
    block_phases = (np.arange(len(result)) + phase) % FRAME_PACKETS
    result[:, 1:] ^= mask[block_phases, :203]
    return result


def _try_packet(rs: OuterReedSolomon, block: np.ndarray) -> bytes | None:
    try:
        packet = rs.decode(block.tobytes())
        parse_packet(packet)
    except ValueError:
        return None
    return packet


def find_energy_phase(
    blocks: np.ndarray, *,
    probe_packets: int = CALIBRATION_PACKETS,
) -> tuple[int, int]:
    """Use actual RS verification, not guessed TS structure, to select PRBS phase."""
    data = np.asarray(blocks, dtype=np.uint8)
    if data.ndim != 2 or data.shape[1] != PACKET_SIZE:
        raise ValueError("expected (N,204) byte-deinterleaved blocks")
    start = BYTE_DEINTERLEAVER_WARMUP
    stop = min(len(data), start + probe_packets)
    if stop - start < 5:
        raise ValueError("too few blocks to identify the PRBS phase")
    rs = OuterReedSolomon()
    scores = []
    for phase in range(FRAME_PACKETS):
        clear = _unmask_blocks(data[start:stop], (phase + start) % FRAME_PACKETS)
        accepted = sum(
            _try_packet(rs, row) is not None
            for row in clear
        )
        scores.append(accepted)
    order = sorted(
        range(FRAME_PACKETS), key=lambda p: (-scores[p], p)
    )
    best, runnerup = order[:2]
    if scores[best] < 3 or scores[best] <= scores[runnerup]:
        raise ValueError(
            f"cannot validate energy-dispersal phase: "
            f"best {scores[best]} RS packets; second {scores[runnerup]}"
        )
    return best, scores[best]


def recover_verified_packets(
    blocks: np.ndarray, phase: int
) -> tuple[list[tuple[int, bytes]], int]:
    """Decode/validate real RS words; drop failures without filling gaps."""
    if np.asarray(blocks).ndim != 2 or np.asarray(blocks).shape[1] != PACKET_SIZE:
        raise ValueError("expected (N,204) data blocks")
    decoded = _unmask_blocks(blocks, phase)
    rs = OuterReedSolomon()
    good = []
    rejected = 0
    for index, block in enumerate(decoded):
        if index < BYTE_DEINTERLEAVER_WARMUP:
            continue
        packet = _try_packet(rs, block)
        if packet is None:
            rejected += 1
        else:
            good.append((index, packet))
    return good, rejected


def recover_file(
    fixture: Path, output: Path, *,
    max_ofdm_symbols: int = 1020,
) -> dict:
    """Validate fixture provenance and atomically write partial real TS."""
    fixture, output = Path(fixture), Path(output)
    metadata_path = fixture.with_suffix(fixture.suffix + ".json")
    target_info = output.with_suffix(output.suffix + ".json")
    partial = output.with_name(output.name + ".partial")
    if output.suffix.lower() != ".ts":
        raise ValueError("output must end in .ts")
    if output.exists() or target_info.exists() or partial.exists():
        raise FileExistsError("will not overwrite existing TS/sidecar/partial")
    if max_ofdm_symbols < 204:
        raise ValueError("--max-ofdm-symbols must be >= 204")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if any((
        metadata.get("mode") != 3,
        metadata.get("modulation") != "QPSK",
        metadata.get("code_rate") != "2/3",
        metadata.get("segments") != 1,
    )):
        raise ValueError("requires Mode-3 QPSK 2/3 one-segment fixture")
    starts = metadata.get("verified_tmcc_frame_starts_original_fft_rows", [])
    if not starts or not all(isinstance(v, int) and v >= 0 for v in starts):
        raise ValueError("requires validated TMCC frame starts")
    first_row = min(starts)
    with np.load(fixture, allow_pickle=False) as archive:
        metrics = archive["qpsk_soft_metric_proxy"]
        original_rows = int(metadata["output_data_symbol_rows"])
        expected_bits = 2 * (384 * original_rows - 120)
        if len(metrics) != expected_bits or metrics.ndim != 1:
            raise ValueError("QPSK proxy and metadata dimensions disagree")
        if not np.isfinite(metrics).all():
            raise ValueError("bad QPSK soft metrics")
    available = original_rows - first_row
    if available < 204:
        raise ValueError("less than one frame remains after TMCC alignment")
    rows = min(max_ofdm_symbols, available)
    coded_start = first_row * 384 * 2
    coded_end = min((first_row + rows) * 384 * 2, len(metrics))
    coded = metrics[coded_start:coded_end]
    # Map signed QPSK axes (positive bit 0) into the [0,1]
    # representation expected by the repository Viterbi decoder.
    uncalibrated_soft = 0.5 - 0.49 * np.tanh(coded)
    usable = (len(uncalibrated_soft) // 3) * 3
    if usable < 204 * 384 * 2:
        raise ValueError("too few whole 2/3-punctured QPSK periods")
    decoded_bits = decode_soft_bits(
        uncalibrated_soft[:usable], "2/3"
    )
    decoded_bytes = np.packbits(
        decoded_bits[:len(decoded_bits) // 8 * 8]
    )
    sync_phase, sync_hits = sync_byte_phase(decoded_bytes)
    blocks = byte_deinterleaved_blocks(decoded_bytes, sync_phase)
    prbs_phase, calibration = find_energy_phase(blocks)
    recovered, rejected = recover_verified_packets(blocks, prbs_phase)
    if len(recovered) < 3:
        raise ValueError("fewer than three RS-validated MPEG-TS packets")
    stats = TransportStats()
    for _, packet in recovered:
        stats.accept(packet)
    pids = Counter(parse_packet(packet).pid for _, packet in recovered)
    details = {
        "source_npz": str(fixture),
        "source_metadata": str(metadata_path),
        "source_verified_tmcc_starts": starts,
        "source_first_fft_row": first_row,
        "processed_ofdm_symbols": rows,
        "rate": "2/3",
        "qpsk_soft_metrics_are_calibrated_llr": False,
        "decoded_inner_fec_bytes": len(decoded_bytes),
        "viterbi_output_sync_phase": sync_phase,
        "first_window_sync_hits": sync_hits,
        "byte_deinterleaver_startup_blocks_discarded": BYTE_DEINTERLEAVER_WARMUP,
        "prbs_phase_within_64_packets": prbs_phase,
        "rs_verified_calibration_packets": calibration,
        "rs_and_ts_accepted_packets": len(recovered),
        "rejected_rs_or_invalid_ts_packets": rejected,
        "total_candidate_204_blocks": len(blocks),
        "accepted_input_block_indices": [i for i, _ in recovered],
        "ts_bytes": len(recovered) * 188,
        "pid_packet_counts": {str(pid): count for pid, count in sorted(pids.items())},
        "pat_programs": stats.pat,
        "pmt_elementary_streams": stats.elementary_streams,
        "continuity_errors_due_to_missing_or_bad_packets": stats.continuity_errors,
        "incomplete_partial_capture": True,
        "sample_continuity_verified": bool(
            metadata.get("sample_continuity_verified", False)
        ),
        "no_missing_packets_synthesized": True,
        "real_mpeg_ts_packets_recovered": True,
        "live_tuner_integration": False,
        "playback_not_guaranteed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TransportWriter(partial) as writer:
            for _, packet in recovered:
                writer.write(packet)
        target_info.write_text(
            json.dumps(details, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        if not output.exists():
            target_info.unlink(missing_ok=True)
        raise
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ch20_partial.ts"))
    parser.add_argument(
        "--max-ofdm-symbols", type=int, default=1020,
        help="process at most N time-deinterleaved rows (>=204; default 1020)",
    )
    args = parser.parse_args()
    try:
        summary = recover_file(
            args.fixture, args.output, max_ofdm_symbols=args.max_ofdm_symbols
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Partial TS recovery failed: {exc}\n")
    print(
        f"REAL RS-verified MPEG-TS packets: "
        f"{summary['rs_and_ts_accepted_packets']}; "
        f"bad/missing: {summary['rejected_rs_or_invalid_ts_packets']}; "
        f"PRBS phase: {summary['prbs_phase_within_64_packets']}/64."
    )
    print(
        f"Saved PARTIAL (possibly non-playable) TS: {args.output}; "
        f"PAT present: {bool(summary['pat_programs'])}. "
        "No live tuner integration; no synthetic packets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

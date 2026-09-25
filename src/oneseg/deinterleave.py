"""Offline ISDB-T Mode-3 Layer-A QPSK carrier and bit deinterleaving.

Standard ARIB STD-B31 §3.9.3 and §3.11:
https://paperzz.com/doc/8177651/arib-std-%E2%80%93-b31
The 384-element carrier allocation is the *standard's normative data
table 3-13(c)*, not copied program source/algorithm from GNU Radio.

One-segment partial-reception layer A does NOT use intersegment
interleaving, and its carrier rotation number is zero. This currently
handles a 384-carrier Mode-3 QPSK single segment only.

No Viterbi, byte deinterleaver, RS codewords, MPEG-TS or video.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PAYLOAD_CARRIERS = 384
QPSK_BIT_DELAY_CARRIERS = 120
FREQUENCY_PERMUTATION = np.array([
    62, 13, 371, 11, 285, 336, 365, 220, 226, 92, 56, 46, 120, 175, 298, 352, 172, 235, 53, 164, 368, 187, 125, 82,
    5, 45, 173, 258, 135, 182, 141, 273, 126, 264, 286, 88, 233, 61, 249, 367, 310, 179, 155, 57, 123, 208, 14, 227,
    100, 311, 205, 79, 184, 185, 328, 77, 115, 277, 112, 20, 199, 178, 143, 152, 215, 204, 139, 234, 358, 192, 309, 183,
    81, 129, 256, 314, 101, 43, 97, 324, 142, 157, 90, 214, 102, 29, 303, 363, 261, 31, 22, 52, 305, 301, 293, 177,
    116, 296, 85, 196, 191, 114, 58, 198, 16, 167, 145, 119, 245, 113, 295, 193, 232, 17, 108, 283, 246, 64, 237, 189,
    128, 373, 302, 320, 239, 335, 356, 39, 347, 351, 73, 158, 276, 243, 99, 38, 287, 3, 330, 153, 315, 117, 289, 213,
    210, 149, 383, 337, 339, 151, 241, 321, 217, 30, 334, 161, 322, 49, 176, 359, 12, 346, 60, 28, 229, 265, 288, 225,
    382, 59, 181, 170, 319, 341, 86, 251, 133, 344, 361, 109, 44, 369, 268, 257, 323, 55, 317, 381, 121, 360, 260, 275,
    190, 19, 63, 18, 248, 9, 240, 211, 150, 230, 332, 231, 71, 255, 350, 355, 83, 87, 154, 218, 138, 269, 348, 130,
    160, 278, 377, 216, 236, 308, 223, 254, 25, 98, 300, 201, 137, 219, 36, 325, 124, 66, 353, 169, 21, 35, 107, 50,
    106, 333, 326, 262, 252, 271, 263, 372, 136, 0, 366, 206, 159, 122, 188, 6, 284, 96, 26, 200, 197, 186, 345, 340,
    349, 103, 84, 228, 212, 2, 67, 318, 1, 74, 342, 166, 194, 33, 68, 267, 111, 118, 140, 195, 105, 202, 291, 259,
    23, 171, 65, 281, 24, 165, 8, 94, 222, 331, 34, 238, 364, 376, 266, 89, 80, 253, 163, 280, 247, 4, 362, 379,
    290, 279, 54, 78, 180, 72, 316, 282, 131, 207, 343, 370, 306, 221, 132, 7, 148, 299, 168, 224, 48, 47, 357, 313,
    75, 104, 70, 147, 40, 110, 374, 69, 146, 37, 375, 354, 174, 41, 32, 304, 307, 312, 15, 272, 134, 242, 203, 209,
    380, 162, 297, 327, 10, 93, 42, 250, 156, 338, 292, 144, 378, 294, 329, 127, 270, 76, 95, 91, 244, 274, 27, 51,
], dtype=np.intp)

if len(FREQUENCY_PERMUTATION) != 384 or not np.array_equal(
    np.sort(FREQUENCY_PERMUTATION), np.arange(384)
):
    raise RuntimeError("Invalid ARIB STD-B31 Mode-3 carrier permutation")


def frequency_deinterleave(payload: np.ndarray) -> np.ndarray:
    """Map received positions 'after[table[i]]' back to original i."""
    data = np.asarray(payload)
    if data.ndim != 2 or data.shape[1] != PAYLOAD_CARRIERS:
        raise ValueError("expected (OFDM symbols, 384 carriers)")
    return data[:, FREQUENCY_PERMUTATION].astype(np.complex64, copy=False)


def time_deinterleave(
    payload: np.ndarray, interleaving_length: int
) -> np.ndarray:
    """Invert Mode-3 convolutional time interleaver, discarding 95*I warmup.

    For carrier i, receiver delay is I*(95-((5*i)%96)).
    Output symbol t represents broadcast data t, after total receiver
    + transmitter buffering delay 95*I. No packet/frame byte alignment
    is implied by the returned first row.
    """
    data = np.asarray(payload)
    if data.ndim != 2 or data.shape[1] != PAYLOAD_CARRIERS:
        raise ValueError("expected (OFDM symbols, 384 carriers)")
    if interleaving_length not in (0, 1, 2, 4):
        raise ValueError("Mode-3 I must be 0, 1, 2 or 4")
    warmup = 95 * interleaving_length
    if len(data) <= warmup:
        raise ValueError(
            f"need more than {warmup} OFDM symbols for Mode-3 I={interleaving_length}"
        )
    delays = interleaving_length * (
        95 - ((5 * np.arange(PAYLOAD_CARRIERS)) % 96)
    )
    row = np.arange(warmup, len(data), dtype=np.intp)[:, None]
    col = np.arange(PAYLOAD_CARRIERS, dtype=np.intp)[None, :]
    return data[row - delays[None, :], col].astype(
        np.complex64, copy=False
    )


def qpsk_bit_deinterleave(
    symbols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo QPSK b1 120-carrier delay; b0=I axis, b1=Q axis.

    Input is a 2D time/frequency-deinterleaved carrier stream. Output
    bits have original serial b0,b1 order. The soft output is a signed
    reliability *proxy* (not calibrated log-likelihood ratios).
    """
    data = np.asarray(symbols)
    if data.ndim != 2 or data.shape[1] != PAYLOAD_CARRIERS:
        raise ValueError("expected (OFDM symbols, 384 QPSK carriers)")
    flat = data.reshape(-1)
    if len(flat) <= QPSK_BIT_DELAY_CARRIERS:
        raise ValueError("need more than 120 carrier symbols")
    proxy = np.empty(
        (len(flat) - QPSK_BIT_DELAY_CARRIERS, 2), dtype=np.float32
    )
    proxy[:, 0] = np.clip(
        flat[:-QPSK_BIT_DELAY_CARRIERS].real, -6.0, 6.0
    )
    proxy[:, 1] = np.clip(
        flat[QPSK_BIT_DELAY_CARRIERS:].imag, -6.0, 6.0
    )
    if not np.isfinite(proxy).all():
        raise ValueError("non-finite QPSK values")
    bits = (proxy < 0).astype(np.uint8)
    return bits.reshape(-1), proxy.reshape(-1)


def process_fixture(
    fixture: Path, tmcc_report: Path, output: Path
) -> dict:
    """Check Mode-3/QPSK/one-segment protected metadata and save stages."""
    fixture = Path(fixture)
    tmcc_report = Path(tmcc_report)
    output = Path(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("output must end in .npz")
    metadata_file = output.with_suffix(output.suffix + ".json")
    if output.exists() or metadata_file.exists():
        raise FileExistsError("output or matching JSON report already exists")
    report = json.loads(tmcc_report.read_text(encoding="utf-8"))
    passed = report.get("bch_parity_verified_frames", [])
    if not passed:
        raise ValueError("no parity-verified TMCC frames")
    layers = [frame["layer_A"] for frame in passed]
    if any(
        layer != layers[0] for layer in layers
    ) or any(
        frame.get("syndrome") != 0 for frame in passed
    ):
        raise ValueError("TMCC verified parameters disagree")
    layer = layers[0]
    if (
        not report.get("tmcc_bch_verified")
        or not report.get("partial_reception_flag", True)
        and not passed[0].get("partial_reception_flag")
        or layer.get("modulation") != "QPSK"
        or layer.get("code_rate") != "2/3"
        or layer.get("segments") != 1
        or layer.get("time_interleaving_mode3") not in (0, 1, 2, 4)
    ):
        raise ValueError("only TMCC-verified Mode-3 Layer-A QPSK 2/3 1seg is supported")
    if report.get("mode_assumed") != 3:
        raise ValueError("requires Mode-3 synchronization report")

    with np.load(fixture, allow_pickle=False) as archive:
        if "equalized_payload_carriers" not in archive:
            raise ValueError("missing equalized carrier array")
        raw = archive["equalized_payload_carriers"]
        starts = archive["verified_tmcc_frame_start_fft_rows"]
        if raw.ndim != 2 or raw.shape[1] != PAYLOAD_CARRIERS:
            raise ValueError("expected 384 carriers per OFDM symbol")
        if int(report["fft_symbols"]) != len(raw):
            raise ValueError("JSON and carrier fixture symbol counts disagree")
        if not np.isfinite(raw).all():
            raise ValueError("fixture has nonfinite constellation values")
        expected_starts = sorted({
            int(x["frame_start_bit_index"]) + 1 for x in passed
            if int(x["frame_start_bit_index"]) + 1 + 204 <= len(raw)
        })
        if starts.tolist() != expected_starts:
            raise ValueError("TMCC report and carrier fixture frame offsets disagree")
    after_freq = frequency_deinterleave(raw)
    i_param = int(layer["time_interleaving_mode3"])
    after_time = time_deinterleave(after_freq, i_param)
    hard, soft = qpsk_bit_deinterleave(after_time)
    if not np.isfinite(after_time).all():
        raise ValueError("invalid deinterleaved constellation")
    temp = output.with_name(output.name + ".partial")
    if temp.exists():
        raise FileExistsError("incomplete output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "source_npz": str(fixture),
        "source_tmcc_report": str(tmcc_report),
        "mode": 3, "partial_reception": True,
        "modulation": "QPSK", "code_rate": "2/3",
        "segments": 1, "time_interleaving_parameter": i_param,
        "time_deinterleaver_warmup_symbols": 95 * i_param,
        "source_fft_symbols": len(raw),
        "output_data_symbol_rows": len(after_time),
        "qpsk_bit_delay_carriers": QPSK_BIT_DELAY_CARRIERS,
        "output_bit_count": len(hard),
        "verified_tmcc_frame_starts_original_fft_rows": expected_starts,
        "sample_continuity_verified": False,
        "ofdm_frame_byte_alignment_verified": False,
        "bit_metrics_are_calibrated_llr": False,
        "viterbi_decoded": False, "mpeg_ts_recovered": False,
    }
    try:
        with temp.open("xb") as destination:
            np.savez_compressed(
                destination,
                deinterleaved_qpsk_carriers=after_time,
                qpsk_hard_bits=hard,
                qpsk_soft_metric_proxy=soft,
                stage="frequency/time/QPSK-bit deinterleaved; NOT FEC / MPEG-TS",
            )
        metadata_file.write_text(
            json.dumps(details, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, output)
    except Exception:
        temp.unlink(missing_ok=True)
        if not output.exists():
            metadata_file.unlink(missing_ok=True)
        raise
    return details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="Layer-A .npz fixture")
    parser.add_argument("--tmcc-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ch20_deinterleaved.npz"))
    args = parser.parse_args()
    try:
        result = process_fixture(args.fixture, args.tmcc_report, args.output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Layer-A deinterleaver failed: {exc}\n")
    print(
        f"{result['source_fft_symbols']} input symbols → "
        f"{result['output_data_symbol_rows']} deinterleaved symbols; "
        f"{result['output_bit_count']:,} QPSK hard/soft bits. "
        "NOT Viterbi decoded; NOT MPEG-TS."
    )
    print(f"Saved: {args.output} plus {args.output}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

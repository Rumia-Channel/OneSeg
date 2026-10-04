"""Synthetic carrier-to-TS regression, not evidence of real RF reception.

The transmitter below uses forward convolutional/byte/time/bit interleavers,
polynomial division, and reedsolo encoding directly. It never calls a receiver
inverse or the production FEC test encoders. The normative frequency-allocation
table is shared as data, so this does not independently certify that ARIB table.
"""
from collections import deque
import hashlib
import json

import numpy as np
import pytest
from reedsolo import RSCodec

from oneseg.deinterleave import FREQUENCY_PERMUTATION, process_fixture
from oneseg.recover import recover_file


def encoded_fixture(tmp_path, *, depth=4, first_frame=17, data_rows=426,
                    corrupt_packets=()):
    # Crop a continuous synthetic transmitter 187 rows into an OFDM frame.
    # Its next real 204-symbol frame then starts at capture row 17. This
    # catches accidental subtraction of the 95*I time-deinterleaver delay.
    capture_start = 204 - first_frame
    capture_rows = data_rows + 95 * depth
    tx_rows = capture_start + capture_rows
    packet_count = (tx_rows * 64 + 203) // 204
    codec = RSCodec(16, nsize=255, fcr=0, prim=0x11D)
    packets = []
    scrambled = bytearray()
    register = 0xA9
    rng = np.random.default_rng(1729)
    for number in range(packet_count):
        packet = (
            bytes((0x47, 0x01, 0x01, 0x10 | (number % 16)))
            + number.to_bytes(4, "big") + rng.bytes(180)
        )
        packets.append(packet)
        word = bytearray(codec.encode(packet))
        # An outer-code error inside the inner-code encoder cannot be repaired
        # by Viterbi. Eight bytes must be corrected; nine must be rejected.
        for packet_number, errors in corrupt_packets:
            if number == packet_number:
                for offset in range(errors):
                    word[20 + offset] ^= 0xA5
        if number % 64 == 0:
            register = 0xA9
        for position in range(204):
            mask = 0
            for _ in range(8):
                feedback = ((register >> 14) ^ (register >> 13)) & 1
                register = ((register << 1) | feedback) & 0x7FFF
                mask = (mask << 1) | feedback
            if position < 203:
                word[position + 1] ^= mask
        scrambled.extend(word[1:] + word[:1])

    queues = [deque([0] * (17 * branch)) for branch in range(12)]
    byte_interleaved = bytearray()
    for index, value in enumerate(scrambled):
        queue = queues[index % 12]
        queue.append(value)
        byte_interleaved.append(queue.popleft())

    coded = []
    state = 0
    for clock, bit in enumerate(np.unpackbits(np.frombuffer(byte_interleaved, np.uint8))):
        state = ((state << 1) | int(bit)) & 0x7F
        # K=7 X/Y taps; punctured rate 2/3 sends X0,Y0,Y1.
        if clock % 2 == 0:
            coded.append((state & 0x4F).bit_count() % 2)
        coded.append((state & 0x6D).bit_count() % 2)
    bits = np.asarray(coded[:tx_rows * 768], dtype=np.float32).reshape(-1, 2)
    qpsk = np.empty(len(bits), dtype=np.complex64)
    qpsk.real = 1 - 2 * bits[:, 0]
    qpsk.imag = np.r_[np.zeros(120), 1 - 2 * bits[:-120, 1]]
    qpsk = qpsk.reshape(tx_rows, 384)
    time_interleaved = np.zeros_like(qpsk)
    for carrier in range(384):
        delay = depth * ((5 * carrier) % 96)
        time_interleaved[delay:, carrier] = qpsk[:tx_rows - delay, carrier]
    rf = np.empty_like(time_interleaved)
    rf[:, FREQUENCY_PERMUTATION] = time_interleaved
    rf = rf[capture_start:]
    starts = list(range(first_frame, capture_rows - 203, 204))

    # Generate TMCC parity by forward polynomial division, independently of
    # oneseg.tmcc. All report fields match real pilot-analysis report format.
    info_bits = [0] * 102
    info_bits[27 - 20] = 1
    layer = "001001" + {0: "000", 1: "001", 2: "010", 4: "011"}[depth] + "0001"
    info_bits[28 - 20:41 - 20] = [int(bit) for bit in layer]
    info = int("".join(map(str, info_bits)), 2)
    polynomial = sum(1 << power for power in (
        82, 77, 76, 71, 67, 66, 56, 52, 48, 40, 36, 34, 24, 22, 18, 10, 4, 0
    ))
    remainder = info << 82
    for bit in range(183, 81, -1):
        if remainder & (1 << bit):
            remainder ^= polynomial << (bit - 82)
    report = {
        "mode_assumed": 3, "fft_symbols": capture_rows, "tmcc_bch_verified": True,
        "bch_parity_verified_frames": [{
            "frame_start_bit_index": start - 1, "syndrome": 0,
            "bch_parity_verified": True, "partial_reception_flag": True,
            "info_bits_hex": hex(info), "parity_bits_hex": hex(remainder),
            "layer_A": {"modulation": "QPSK", "code_rate": "2/3",
                        "segments": 1, "time_interleaving_mode3": depth},
        } for start in starts],
    }
    source = tmp_path / "synthetic_carriers.npz"
    np.savez(source, equalized_payload_carriers=rf,
             verified_tmcc_frame_start_fft_rows=np.asarray(starts, dtype=np.int32))
    tmcc = tmp_path / "synthetic_tmcc.json"
    tmcc.write_text(json.dumps(report), encoding="utf-8")
    return source, tmcc, packets


@pytest.mark.parametrize("depth,first_frame", [(0, 1), (1, 17), (2, 139), (4, 17)])
def test_independently_encoded_carriers_recover_original_packets(tmp_path, depth, first_frame):
    source, tmcc, packets = encoded_fixture(
        tmp_path, depth=depth, first_frame=first_frame,
        data_rows=first_frame + 204, corrupt_packets=((75, 8), (80, 9)),
    )
    bits = tmp_path / "bits.npz"
    stage = process_fixture(source, tmcc, bits, compress=False)
    assert stage["tmcc_protected_fields_reverified"] is True
    assert stage["fixture_sha256"] == hashlib.sha256(bits.read_bytes()).hexdigest()
    output = tmp_path / "received.ts"
    result = recover_file(bits, output, max_ofdm_symbols=204)
    received = output.read_bytes()
    lookup = {packet: index for index, packet in enumerate(packets)}
    indices = [lookup[received[offset:offset + 188]] for offset in range(0, len(received), 188)]
    assert 75 in indices  # eight erroneous outer bytes really were corrected
    assert 80 not in indices  # uncorrectable word was never synthesized
    assert indices == [index for index in range(indices[0], indices[-1] + 1) if index != 80]
    assert len(indices) > 40
    assert result["rejected_rs_or_invalid_ts_packets"] == 1
    assert result["source_first_fft_row"] == first_frame
    assert result["source_fixture_sha256_verified"] is True
    assert result["source_tmcc_protected_fields_reverified"] is True
    assert result["trailing_qpsk_coded_bits_unavailable"] == 240
    assert result["processed_coded_bits"] == 204 * 768 - 240
    assert result["no_missing_packets_synthesized"] is True
    assert result["live_tuner_integration"] is False


def test_recovery_rejects_mismatched_fixture_sidecar(tmp_path):
    source, tmcc, _ = encoded_fixture(tmp_path, depth=0)
    bits = tmp_path / "bits.npz"
    process_fixture(source, tmcc, bits)
    with np.load(bits) as archive:
        metrics = archive["qpsk_soft_metric_proxy"].copy()
    metrics[0] *= -1
    np.savez(bits, qpsk_soft_metric_proxy=metrics)
    with pytest.raises(ValueError, match="SHA-256"):
        recover_file(bits, tmp_path / "never.ts")
    assert not (tmp_path / "never.ts").exists()


@pytest.mark.parametrize("mutation,message", [
    (lambda report: report["bch_parity_verified_frames"][0].update(parity_bits_hex="0x0"), "parity check"),
    (lambda report: report["bch_parity_verified_frames"][0]["layer_A"].update(code_rate="1/2"), "protected bits"),
    (lambda report: report["bch_parity_verified_frames"][0].update(frame_start_bit_index=-1), "offsets"),
    (lambda report: report.update(tmcc_bch_verified="yes"), "TMCC-verified"),
])
def test_forged_tmcc_claims_fail_before_output(tmp_path, mutation, message):
    source, tmcc, _ = encoded_fixture(tmp_path, depth=0)
    report = json.loads(tmcc.read_text())
    mutation(report)
    tmcc.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        process_fixture(source, tmcc, tmp_path / "never.npz")
    assert not (tmp_path / "never.npz").exists()


def synthetic_iq(source, tmcc, output, *, corrupt_tmcc=False):
    """Forward central-segment OFDM at 2.048 MS/s, with a 73 Hz CFO.

    Uses literal normative SP/AC/TMCC locations and an independent 11-bit
    pilot PRBS. No receiver FFT/extraction/equalizer helper is used here.
    """
    from scipy.signal import resample_poly

    with np.load(source) as archive:
        payload = archive["equalized_payload_carriers"]
    frame = json.loads(tmcc.read_text())["bch_parity_verified_frames"][0]
    info, parity = int(frame["info_bits_hex"], 16), int(frame["parity_bits_hex"], 16)
    protected = np.asarray([int(bit) for bit in f"{info:0102b}{parity:082b}"], np.uint8)
    if corrupt_tmcc:
        protected[-1] ^= 1
    state = 0x7FF
    polarities = []
    for _ in range(3025):
        polarities.append(1 - 2 * (state & 1))
        state = (state >> 1) | ((((state >> 2) ^ state) & 1) << 10)
    polarities = np.asarray(polarities[2592:]) * (4 / 3)
    fft = np.zeros((len(payload), 1024), dtype=np.complex64)
    level = 1.0
    even_sync = [int(bit) for bit in "0011010111101110"]
    for row in range(len(payload)):
        sp = np.arange(3 * ((row + 187) % 4), 432, 12)
        position, frame_number = (row - 17) % 204, (row - 17) // 204
        if 1 <= position <= 16:
            bit = even_sync[position - 1] ^ (frame_number % 2)
        elif position >= 20:
            bit = int(protected[position - 20])
        else:
            bit = 0
        level *= 1 - 2 * bit
        segment = np.ones(433, dtype=np.complex64)
        segment[sp] = polarities[sp]
        segment[[101, 131, 286, 349]] = level * (4 / 3)
        segment[432] = polarities[432]
        excluded = set(sp) | {101, 131, 286, 349, 7, 89, 206, 209, 226, 244, 377, 407}
        positions = [carrier for carrier in range(432) if carrier not in excluded]
        segment[positions] = payload[row]
        fft[row, 296:729] = segment
    waveform = np.fft.ifft(np.fft.ifftshift(fft, axes=1), axis=1)
    waveform = np.concatenate([waveform[:, -128:], waveform], axis=1).ravel()
    waveform = resample_poly(waveform, 252, 125) * 4
    waveform *= np.exp(2j * np.pi * 73 * np.arange(len(waveform)) / 2_048_000)
    np.asarray(waveform, dtype="<c8").tofile(output)
    output.with_suffix(".c64.json").write_text(json.dumps({
        "sample_rate_hz": 2_048_000, "format": "complex64",
        "center_frequency_hz": 515_142_857,
        "source": "synthetic test waveform; NOT real RF reception",
    }))


@pytest.mark.parametrize("corrupt_tmcc", [False, True])
def test_full_synthetic_iq_to_rs_validated_transport(tmp_path, corrupt_tmcc):
    from oneseg.decode import UnverifiedTmccError, decode_capture

    source, tmcc, packets = encoded_fixture(tmp_path, depth=4, first_frame=17, data_rows=426)
    capture = tmp_path / "synthetic.c64"
    synthetic_iq(source, tmcc, capture, corrupt_tmcc=corrupt_tmcc)
    output = tmp_path / "full_pipeline.ts"
    if corrupt_tmcc:
        with pytest.raises(UnverifiedTmccError):
            decode_capture(capture, output, seconds=1.2, max_ofdm_symbols=204)
        assert not output.exists()
        assert not output.with_suffix(".ts.json").exists()
        return
    result = decode_capture(capture, output, seconds=1.2, max_ofdm_symbols=204)
    assert output.read_bytes() == b"".join(packets[64:117])
    assert result["rs_and_ts_accepted_packets"] == 53
    assert result["rejected_rs_or_invalid_ts_packets"] == 0
    assert result["tmcc_parity_verified_frames"] == 3
    assert result["source_tmcc_protected_fields_reverified"] is True
    assert result["fractional_cfo_hz"] == pytest.approx(73, abs=1)
    assert result["input_cp_quality"] > 0.98
    assert result["input_pilot_coherence"] > 0.98
    assert result["not_live_reception"] is True


def test_seeded_noise_cannot_publish_transport(tmp_path):
    from oneseg.decode import UnverifiedTmccError, decode_capture

    rng = np.random.default_rng(20261004)
    count = int(0.6 * 2_048_000)
    noise = (rng.standard_normal(count) + 1j * rng.standard_normal(count)) * 0.05
    capture = tmp_path / "synthetic_noise.c64"
    np.asarray(noise, dtype="<c8").tofile(capture)
    capture.with_suffix(".c64.json").write_text(json.dumps({
        "sample_rate_hz": 2_048_000, "format": "complex64",
        "center_frequency_hz": 515_142_857,
        "source": "seeded synthetic noise; NOT RF reception",
    }))
    output = tmp_path / "never.ts"
    with pytest.raises(UnverifiedTmccError):
        decode_capture(capture, output, seconds=0.6, max_ofdm_symbols=204)
    assert not output.exists()
    assert not output.with_suffix(".ts.json").exists()

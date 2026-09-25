import json

import numpy as np
import pytest

from oneseg.deinterleave import (
    FREQUENCY_PERMUTATION,
    frequency_deinterleave,
    time_deinterleave,
    qpsk_bit_deinterleave,
    process_fixture,
)


def test_standard_mode3_mapping_is_exact_permutation():
    assert len(FREQUENCY_PERMUTATION) == 384
    np.testing.assert_array_equal(np.sort(FREQUENCY_PERMUTATION), np.arange(384))
    assert FREQUENCY_PERMUTATION[:4].tolist() == [62, 13, 371, 11]
    assert FREQUENCY_PERMUTATION[-4:].tolist() == [244, 274, 27, 51]
    source = np.arange(384, dtype=np.complex64)
    mapped = np.empty(384, dtype=np.complex64)
    mapped[FREQUENCY_PERMUTATION] = source
    np.testing.assert_array_equal(
        frequency_deinterleave(mapped[None, :])[0], source
    )


def test_mode3_time_interleave_roundtrip_after_warmup():
    i_param = 4
    n = 800
    original = (
        np.arange(n, dtype=np.float32)[:, None]
        + (np.arange(384, dtype=np.float32) / 1000)[None, :]
    ).astype(np.complex64)
    mapped = np.zeros_like(original)
    delays = i_param * ((5 * np.arange(384)) % 96)
    for col, d in enumerate(delays):
        mapped[d:, col] = original[:n-d, col]
    restored = time_deinterleave(mapped, i_param)
    np.testing.assert_array_equal(restored, original[:n-95*i_param])
    assert time_deinterleave(original, 0).shape == original.shape
    with pytest.raises(ValueError):
        time_deinterleave(original[:380], 4)


def test_qpsk_receiver_delays_b0_by_120_carriers():
    original = (np.arange(384*3) % 2 * 2 - 1).astype(np.float32)
    b0 = original
    b1 = -b0
    mapped = np.empty((len(original),), dtype=np.complex64)
    mapped.real = b0
    mapped.imag = np.r_[np.zeros(120, dtype=np.float32), b1[:-120]]
    bits, proxy = qpsk_bit_deinterleave(mapped.reshape(3, 384))
    np.testing.assert_array_equal(
        bits.reshape(-1, 2)[:, 0], (b0[:-120] < 0).astype(np.uint8)
    )
    np.testing.assert_array_equal(
        bits.reshape(-1, 2)[:, 1], (b1[:-120] < 0).astype(np.uint8)
    )
    assert len(proxy) == 2 * (len(b0) - 120)


def test_requires_report_matching_fixture_and_safe_output(tmp_path):
    npz = tmp_path / "layer_a.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(
            stream,
            equalized_payload_carriers=np.ones((600,384), dtype=np.complex64),
            verified_tmcc_frame_start_fft_rows=np.array([101], dtype=np.int32),
        )
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "mode_assumed": 3,
        "fft_symbols": 600,
        "tmcc_bch_verified": True,
        "bch_parity_verified_frames": [
            {"frame_start_bit_index": 100, "syndrome": 0,
             "partial_reception_flag": True,
             "layer_A": {"modulation": "QPSK", "code_rate":"2/3",
                         "segments":1, "time_interleaving_mode3":4}}
        ]
    }), encoding="utf-8")
    output = tmp_path / "done.npz"
    summary = process_fixture(npz, report, output)
    assert summary["output_data_symbol_rows"] == 220
    assert summary["output_bit_count"] == 2*(220*384-120)
    assert not summary["mpeg_ts_recovered"]
    with np.load(output, allow_pickle=False) as done:
        assert done["deinterleaved_qpsk_carriers"].shape == (220, 384)
        assert done["qpsk_hard_bits"].shape == (summary["output_bit_count"],)
    with pytest.raises(FileExistsError):
        process_fixture(npz, report, output)


def test_metadata_mismatch_is_rejected(tmp_path):
    path = tmp_path/"x.npz"
    np.savez_compressed(path, equalized_payload_carriers=np.ones((600,384)),
                        verified_tmcc_frame_start_fft_rows=[10])
    report = tmp_path/"r.json"
    report.write_text(json.dumps({
        "mode_assumed":3, "fft_symbols":600, "tmcc_bch_verified":True,
        "bch_parity_verified_frames":[
            {"frame_start_bit_index":10,"syndrome":0,
             "partial_reception_flag":True,
             "layer_A":{"modulation":"QPSK","code_rate":"2/3",
                        "segments":1,"time_interleaving_mode3":4}}
        ]
    }))
    with pytest.raises(ValueError, match="frame offsets"):
        process_fixture(path, report, tmp_path/"out.npz")

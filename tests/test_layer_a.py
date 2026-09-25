import numpy as np
import pytest

from oneseg.layer_a import (
    AC1_CARRIERS,
    PAYLOAD_CARRIERS,
    data_carrier_indices,
    extract_layer_a_carriers,
    save_layer_a_fixture,
)
from oneseg.pilots import TMCC_CARRIERS, scattered_pilot_indices


def test_each_symbol_contains_exactly_384_payload_carriers():
    all_ac = set(AC1_CARRIERS.tolist())
    tmcc = set(TMCC_CARRIERS.tolist())
    for phase in range(4):
        indices = data_carrier_indices(phase, 0)
        assert len(indices) == PAYLOAD_CARRIERS
        assert len(set(indices.tolist())) == PAYLOAD_CARRIERS
        assert all(i < 432 for i in indices)
        assert not set(indices.tolist()) & all_ac
        assert not set(indices.tolist()) & tmcc
        assert not set(indices.tolist()) & set(
            scattered_pilot_indices(phase, 0).tolist()
        )
        assert all(
            data_carrier_indices(i, phase).shape == (384,)
            for i in range(4)
        )


def test_payload_order_and_exclusion():
    eq = np.tile(
        np.arange(433, dtype=np.float32), (8, 1)
    ).astype(np.complex64)
    result = extract_layer_a_carriers(eq, pilot_phase=2)
    assert result.shape == (8, 384)
    for n in range(8):
        np.testing.assert_array_equal(
            result[n].real,
            data_carrier_indices(n, 2),
        )


def test_save_only_verified_full_frame_positions(tmp_path):
    out = tmp_path / "data.npz"
    equalized = np.ones((500, 384), dtype=np.complex64)
    frames = [
        {"frame_start_bit_index": 20, "bch_parity_verified": True},
        {"frame_start_bit_index": 224, "bch_parity_verified": True},
        {"frame_start_bit_index": 300, "bch_parity_verified": False},
        {"frame_start_bit_index": 400, "bch_parity_verified": True},
    ]
    info = save_layer_a_fixture(
        out, payload=equalized, pilot_phase=0,
        tmcc_frames=frames, integer_offset_bins=11,
    )
    assert info["verified_tmcc_frame_start_fft_rows"] == [21, 225]
    assert not info["mpeg_ts_recovered"]
    with np.load(out) as saved:
        assert saved["equalized_payload_carriers"].shape == (500, 384)
        np.testing.assert_array_equal(
            saved["verified_tmcc_frame_start_fft_rows"],
            [21, 225],
        )
        assert int(saved["integer_offset_bins"]) == 11
        assert "NOT TS" in str(saved["stage"])
    with pytest.raises(FileExistsError):
        save_layer_a_fixture(
            out, payload=equalized, pilot_phase=0,
            tmcc_frames=frames, integer_offset_bins=11,
        )


def test_invalid_shape_or_phase():
    with pytest.raises(ValueError):
        extract_layer_a_carriers(np.zeros((4, 432)), pilot_phase=0)
    with pytest.raises(ValueError):
        data_carrier_indices(-1)
    with pytest.raises(ValueError):
        data_carrier_indices(0, -1)

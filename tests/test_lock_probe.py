import pytest
from oneseg.lock_probe import summarize_locks

def test_strong_repeating_gi_and_no_false_ts():
    rows=[dict(mode=3,guard="1/8",correlation=x,cfo_hz=-300)
          for x in (.95,.98,.94,.71,.98)]
    got=summarize_locks(rows)
    assert got["repeated_ofdm_cp_candidate"]
    assert got["mode"]==3 and got["guard"]=="1/8"
    assert got["median_correlation"]==pytest.approx(.95)
    assert not got["tmcc_verified"] and not got["mpeg_ts_recovered"]

def test_weak_noise_not_tv_lock():
    rows=[dict(mode=3,guard="1/8",correlation=.2,cfo_hz=0) for _ in range(5)]
    assert not summarize_locks(rows)["repeated_ofdm_cp_candidate"]

"""Multiwindow OFDM cyclic-prefix probe: not a TMCC or MPEG-TS decoder."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt, resample_poly
from .dsp import DEFAULT_SAMPLE_RATE
from .ofdm import find_symbol_lock

def summarize_locks(windows):
    acceptable = [w for w in windows if w["correlation"] >= 0.65]
    top = Counter((w["mode"], w["guard"]) for w in acceptable).most_common(1)
    pair, count = top[0] if top else ((None, None), 0)
    consistent = count >= 3 and count >= (len(windows) + 1) // 2
    same = [w for w in acceptable if (w["mode"], w["guard"]) == pair]
    return {
        "repeated_ofdm_cp_candidate": consistent,
        "mode": pair[0] if consistent else None,
        "guard": pair[1] if consistent else None,
        "agreeing_windows": count,
        "median_correlation": float(np.median([w["correlation"] for w in same])) if same else None,
        "median_fractional_cfo_hz": float(np.median([w["cfo_hz"] for w in same])) if same else None,
        "tmcc_verified": False, "mpeg_ts_recovered": False,
    }

def probe_capture(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if int(meta["sample_rate_hz"]) != DEFAULT_SAMPLE_RATE or meta.get("format") != "complex64":
        raise ValueError("expected 2.048MS/s complex64")
    if path.stat().st_size % 8:
        raise ValueError("truncated complex64 bytes")
    total = path.stat().st_size // 8
    length = DEFAULT_SAMPLE_RATE // 4
    if total < length:
        raise ValueError("capture shorter than 250ms")
    starts = sorted({int((total-length)*fraction) for fraction in (0,.2,.4,.6,.8)})
    filt = butter(8, 205000, fs=DEFAULT_SAMPLE_RATE, output="sos")
    windows = []
    with path.open("rb") as fd:
        for pos in starts:
            fd.seek(pos * 8)
            raw = np.fromfile(fd, dtype="<c8", count=length)
            if len(raw) != length:
                raise ValueError("short capture")
            baseband = resample_poly(sosfilt(filt, raw-np.mean(raw, dtype=np.complex128)),125,252).astype(np.complex64)
            lock = find_symbol_lock(baseband, min_symbols=30)
            windows.append({
                "seconds": round(pos/DEFAULT_SAMPLE_RATE,4),
                "mode": lock.mode, "guard":lock.guard,
                "correlation":round(lock.cp_quality,5),
                "cfo_hz":round(lock.coarse_cfo_hz,2),
            })
    result = {"capture":path.name,"center_frequency_hz":meta["center_frequency_hz"],
              "gain_db":meta.get("gain_db"),"windows":windows}
    result.update(summarize_locks(windows))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture",type=Path)
    parser.add_argument("--json",type=Path,help="optional report path")
    args = parser.parse_args()
    try:
        result=probe_capture(args.capture)
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        parser.exit(2,"CP probe failed: "+str(exc)+"\n")
    for w in result["windows"]:
        print(f"{w['seconds']:.2f}s mode {w['mode']} GI {w['guard']} CP {w['correlation']:.3f} CFO {w['cfo_hz']:.1f}Hz")
    print("Repeated OFDM CP candidate:",result["repeated_ofdm_cp_candidate"],
          "NOT TMCC verified; NOT MPEG-TS or live TV.")
    if args.json:
        args.json.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

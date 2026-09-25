# OneSeg — DS-DT308SV SDR / 1seg research app

Windows 11 / Python 3.12 / **uv only**. No Conda, GNU Radio, WSL or administrator privileges at runtime.

> **Status (prototype):** Windows SDR reception, mono FM, I/Q capture, offline one-seg OFDM/FFT analysis, Viterbi and RS FEC modules, post-inner-FEC MPEG-TS output, and in-app playback of **already decoded** MPEG-TS. **There is still NO working, integrated DS-DT308SV I/Q → one-seg TS decoder.** Do not confuse an RF-tuned physical channel or a TS player with live television reception.

## Installation (PowerShell)

1. Install [uv](https://docs.astral.sh/uv/) and `uv python install 3.12`.
2. With the DS-DT308SV plugged in, use [Zadig](https://zadig.akeo.ie/) **once** to bind the correct RTL2832U interface to WinUSB. Save the existing driver information first. This may prevent the original CHUSEI software from running until you restore its driver.
3. In this repository run:

```powershell
uv sync
uv run oneseg
```

There is no USB device available in CI: reception and Windows sound-device behavior require testing with the actual DS-DT308SV. `pyrtlsdr[lib]` installs bundled `librtlsdr` binaries where supported; if loading fails, check Windows DLL dependencies and matching x64 architectures.

## Features

- **SDR mode:** spectrum, manual frequency tuning, optional wide-FM mono audio (a sound output device is required).
- **1seg research mode:** physical channels 13–52, RF frequency calculation, spectrum, and I/Q recording. It does **not** decode television yet.
- **Record:** choose a `.c64` file (little-endian complex64 I/Q). A `.json` sidecar records center frequency, sample rate, gain and PPM. Retuning stops the current recording to avoid mixing frequencies.
- **Offline diagnostics:** `uv run oneseg-inspect path/to/capture.c64` prints *candidate* cyclic-prefix correlation peaks; `uv run oneseg-ofdm path/to/capture.c64 --output ofdm_symbols.npz` writes central-segment FFT constellations (NOT TS). Shorter guard windows overlap longer ones and false locks are possible.
- **TS backend:** `uv run oneseg-ts rs204 input.rs204 output.ts` converts synthetic or externally recovered *aligned* shortened outer RS(204,188) codewords into checked TS packets. `uv run oneseg-ts postfec input.bin output.ts` accepts **externally recovered, frame-aligned Viterbi output bytes** (not raw I/Q), runs 12-way byte deinterleaving, energy descrambling, RS correction and TS validation. Use `--no-byte-deinterleave` only if the input is already byte-deinterleaved. `uv run oneseg-ts inspect output.ts` prints packet and PAT/PMT statistics. Wrong frame alignment or PRBS state produces no TS.
- **TS viewer:** click “Play decoded TS…” to open an existing .ts with PyAV/FFmpeg for video and audio. This does not receive live television yet.

## Development

```powershell
uv run pytest
uv run oneseg --help
```

Project root: `src/oneseg/`. The GUI owns the worker thread; the RTL device is opened and closed on that thread and commands are consumed between read buffers.

### One-seg decoder milestones

1. Prove I/Q capture and validate frequency/PPM/gain for the actual FC0013 dongle.
2. Narrowband filter, shift and resample to one-segment baseband; verify correct central segment alignment.
3. Detect OFDM mode/guard interval, synchronize symbols and correct carrier/sample-frequency offsets.
4. FFT, pilot-based channel estimation/equalization, TMCC, segment extraction, bit/symbol deinterleaving.
5. Integrate the existing offline depuncturing/Viterbi, byte deinterleaving, energy descrambler, Reed–Solomon and TS backend; establish continuous block boundaries and test independently against known-good ISDB-T I/Q.
6. Validate real over-the-air 188-byte MPEG-TS packets, connect live TS to PyAV, and add a television service list to the GUI.

Reference only: [git-artes/gr-isdbt](https://github.com/git-artes/gr-isdbt) (GPL-3.0). No source code has been copied from it. Review licensing before porting or redistributing third-party decoding code.

## Limitations

- Only one process can open the RTL-SDR at a time; close this app before opening SDR++.
- The frequency dropdown is a **physical RF channel**, not an EPG/program list.
- No warranty of successful RF reception, 1seg decoding, or FM audio playback on any particular Windows/USB configuration.

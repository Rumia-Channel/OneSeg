# OneSeg — DS-DT308SV SDR / 1seg research app

Windows 11 / Python 3.12 / **uv only**. No Conda, GNU Radio, WSL or administrator privileges at runtime.

> **Status (prototype):** SDR spectrum display, RF tuning, optional mono WFM listening, raw I/Q capture, and offline OFDM cyclic-prefix diagnostics. **ISDB-T 1seg video/audio decoding, TS output, and television channel scanning are NOT implemented.** Do not confuse an RF-tuned physical channel with a decoded TV service.

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
- **Offline diagnostics:** `uv run oneseg-inspect path/to/capture.c64` prints *candidate* cyclic-prefix correlation peaks; these are not proof of a valid ISDB-T lock.

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
5. QPSK demap, rate-dependent depuncturing/Viterbi, byte deinterleaving, energy descrambling, Reed–Solomon, TS synchronization.
6. Produce verifiable 188-byte MPEG-TS packets; integrate a decoder and service list into the GUI.

Reference only: [git-artes/gr-isdbt](https://github.com/git-artes/gr-isdbt) (GPL-3.0). No source code has been copied from it. Review licensing before porting or redistributing third-party decoding code.

## Limitations

- Only one process can open the RTL-SDR at a time; close this app before opening SDR++.
- The frequency dropdown is a **physical RF channel**, not an EPG/program list.
- No warranty of successful RF reception, 1seg decoding, or FM audio playback on any particular Windows/USB configuration.

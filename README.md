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

## Windows USB error `LIBUSB_ERROR_NOT_SUPPORTED (-12)`

If the GUI reports `Could not open SDR (device index = 0)`, the error occurs at USB device opening, *before* ISDB-T or RF processing. This is not a uv dependency resolution error. A compatible RTL-SDR was detected, but the USB backend cannot open it.

1. Close this app and all other SDR/TV apps, unplug/replug DS-DT308SV, and try a USB 2.0 port if available.
2. Check **Device Manager → Details → Hardware IDs** for the exact DS-DT308SV USB VID/PID, or open Zadig's Options → List All Devices and identify it by its disappearance when unplugged. Many RTL2832U dongles show `0BDA:2838`, **but verify the actual device, do not assume this ID for every dongle**.
3. In Zadig, select the correct RTL2832U / `Bulk-In, Interface (Interface 0)` entry, **not Interface 1** and not another computer peripheral. Set target driver to WinUSB and install/replace only that interface's driver after checking the ID. If it is already WinUSB, inspect the other interface/parent entries and USB port before blindly reinstalling.
4. Run `uv run oneseg-doctor`. It enumerates compatible devices and attempts a device open/close test without modifying drivers. If another SDR app can open the tuner but this cannot, collect its driver and DLL versions before changing drivers.

Reference: [RTL-SDR guide to usb_open_error -12](https://www.rtl-sdr.com/signalseverywhere-windows-10-usb_open_error-12-fix/) and [libusb Windows drivers](https://github.com/libusb/libusb/wiki/Windows). Changing the driver to WinUSB generally prevents the original vendor tuner app from using the same interface until its driver is restored.

## FC0013 detected, but `Could not set freq. offset to 0 ppm`

If Zadig is installed correctly and the terminal logs `Found Fitipower FC0013 tuner`, USB access is already working. Some `librtlsdr` builds return `LIBUSB_ERROR_INVALID_PARAM (-2)` for a PPM setting equal to the one already in use. On a fresh RTL-SDR open the default is 0 ppm, so requesting 0 is a redundant call, *not a bad USB driver*. The app now skips redundant PPM writes and only invokes `freq_correction` when it changes, including nonzero -> 0.

After `git pull` and `uv sync`, retry `uv run oneseg`. Once the spectrum works, try FM in SDR mode (choose a station available in your region). Live 1seg video is still not integrated.

## Capture a short real-world one-seg IQ fixture (Windows 11, uv only)

A normal SDR screenshot **cannot** establish that the receiver is locked to ISDB-T or that the decoder will work on real broadcasts. Use a **physical channel that actually carries television in your region**, connect a suitable UHF antenna, and close the GUI/SDR++ before capturing.

```powershell
git pull
uv sync
uv run oneseg-capture --channel 27 --seconds 3 --output oneseg_ch27.c64
uv run oneseg-ofdm oneseg_ch27.c64 --seconds 0.5 --output ofdm_ch27.npz
```

Replace 27 with your local station's UHF physical channel. Capture 3 seconds at 2.048 MS/s: 6,144,000 complex64 samples (~49 MB decimal) plus `oneseg_ch27.c64.json` with tuning, gain and PPM. This command opens the tuner, discards one warm-up buffer, records a fixed number of aligned reads, closes the tuner, and never overwrites an existing recording. It does **not** create TV video or `.ts` on its own. Retain both files when reporting failures; a brief screenshot is not a substitute for actual I/Q samples.

This test vector is needed to implement and verify stable central-segment filtering, pilot equalization, TMCC, complete ISDB-T deinterleaving and frame alignment. Prior isolated FEC tests do not validate an over-the-air I/Q-to-TS chain.

### GUI capture shortcut

After starting the receiver, choose a **local active** UHF physical channel (13–52). Click `Capture 3s for decoder…` and choose a new `.c64` filename. The worker automatically stops recording after 3 seconds and writes an adjacent `.c64.json` metadata file. This does **not** decode live television. Send both files together when reporting real-world decoding failures.

## Real capture quality check

`uv run oneseg-quality oneseg_ch27.c64` reports sample count, complex RMS,
and the fraction of samples where either I or Q is at full-scale (±1.0).
Above 5% triggers an *overload warning*; it does not prove which analog gain
stage is overloaded and is not an ISDB-T/TMCC lock result. The GUI now warns
on a high-clipping live block and permits negative FC0013 manual gain values
(-10 to +20 dB input range, tuner rounds to supported steps). Disable
Automatic RF gain and try -9.9 dB first if overloaded; compare captures from
a physically active local UHF channel at different gains.

Reference for reported I/Q full-scale occupancy on the 2026-09-25 ch27
user-supplied 3-second capture: ~49% of samples had at least one full-scale
I/Q component. That recording is not adequate to declare video decoding
functional, and the receiver must not invent MPEG-TS from it.

## UHF physical-channel RF scan (Windows GUI)

Rather than assuming that **27ch** is in use locally, start the RTL-SDR,
switch **Mode** to `1seg RF research`, uncheck **Automatic** RF gain,
and use one fixed manual gain across all channels (e.g. -9.9 dB if the
previous signal overloaded). Click `Scan UHF 13–52 (RF)`.

The receiver tunes the *center* of each 6 MHz UHF physical channel 13–52
using the existing 2.048 MS/s hardware and measures three post-settle
blocks. The GUI reports channel/frequency, DC-corrected complex I/Q power
(dB relative to complex full-scale), power above the **median of all scanned
channels**, and the percent of samples at I or Q full-scale.

- `RF CANDIDATE (NOT TV LOCK)`: at least 6 dB above the **scan-wide**
  median and no >5% clipping. Could be other RF interference; weaker
  valid stations can be missed.
- `OVERLOAD / RETEST`: >5% full-scale I/Q; reduce manual gain and rerun.
- `NO STRONG RF CONTRAST`: no strong difference from other scanned
  centers; does **not** prove there is no broadcast in that channel.

The scan keeps one fixed manual gain, holds the same opened USB device,
shows progress, allows cancellation, and restores the previous tune.
Use `Tune selected` to set the physical channel without retyping it;
then `Capture 3s for decoder…` to save a usable fixture. Export the
measurements using `Export scan CSV…` and share the CSV if useful.

**This is not a decoded station scan**: no TMCC verification, transport
stream, program list or audio/video is produced. At 2.048 MS/s it can
measure the middle part of each 6 MHz channel, not the entire 13-seg signal
at once. The threshold is relative and can fail if the median is
contaminated, all channels have similar energy, or local signals are weak.

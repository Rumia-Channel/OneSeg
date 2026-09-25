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

## 20ch: repeated OFDM cyclic-prefix evidence (2026-09-25)

The actual uploaded 3-second recording at 515,142,857 Hz (20ch), fixed 0.0 dB RF gain and 0 ppm contained 6,144,000 complex64 samples, RMS ~0.2612, and 0% I/Q full-scale hits. Repeating OFDM cyclic-prefix correlations across five spaced 250ms windows strongly favored **mode 3, guard interval 1/8**, with correlations approximately 0.946, 0.983, 0.940, 0.705 and 0.982; fractional frequency offsets were about -305 to -293 Hz. This is a significant improvement over the earlier 27ch capture. Repeated CP is not a TMCC or service lock, and cannot on its own produce playable TS.

To independently reproduce these intermediate measurements (the tuner does not need to be connected):

    uv run oneseg-lock oneseg_ch20_20260925_161517.c64 --json ch20_lock.json

Keep the matching .c64.json alongside the recording. The next unverified step is **pilot-aided equalization and TMCC/frame synchronization** of the central one-segment carriers, followed by byte/bit deinterleaving, FEC and a real MPEG-TS output. Do not claim real television playback from this CP result.


## Pilot alignment, experimental TMCC and capture continuity (2026-09-25)

The 20ch / 0 dB / 0 ppm real recording allowed identifying an additional
uncompensated **integer carrier shift of +12 FFT bins** (approximately
+11.9 kHz at the one-segment processing sample rate). In 30 early
symbols the expected pilot PRBS pattern has ~0.946 adjacent-pilot
channel coherence at this shift; no other tested shift in ±30 came
close. The fractional cyclic-prefix CFO estimate was approximately
-305 Hz. Both must be corrected before attempting to read TMCC
carriers at their nominal positions. This is **not** the same as a
verified tuner PPM calibration; RF center error and oscillator error
cannot yet be separated.

The uploaded recording also shows substantial changes in recovered
OFDM timing and scattered-pilot phase across its ~131072-sample
synchronous read boundaries. This is **consistent with missing samples**
between separate synchronous USB reads, but cannot conclusively
distinguish USB buffering, receiver processing, and other capture
discontinuities without hardware timestamps. An earlier statement that
the original 3-second capture was sufficient for full TMCC decoding
was overconfident: contiguous samples must be confirmed across the
whole 204-symbol ISDB-T frame. Do not concatenate out-of-sync
partial captures and claim live decoding.

The **3-second GUI decoder capture button** and \`oneseg-capture\` CLI
now use the same USB sync-reader / separate writer-thread path,
write exact complex64 sample counts, and atomically publish the
recording and metadata with `"acquisition": "sync_usb_reader_threaded_iq_writer"`.
While the GUI captures, spectrum updates are paused to reduce CPU
and USB-buffer latency. The manual \`Record I/Q…\` still uses
synchronous reads and is **not** recommended for decoder fixtures.

After \`git pull\` and \`uv sync\`, with the GUI/SDR++ closed, capture a
new 20ch test vector with 0 dB manual gain:

\`\`\`powershell
uv run oneseg-capture --channel 20 --seconds 3 --gain 0 --ppm 0 --output ch20_async.c64
uv run oneseg-quality ch20_async.c64
uv run oneseg-lock ch20_async.c64
uv run oneseg-pilots ch20_async.c64 --seconds 1.2 --json ch20_pilots.json
\`\`\`

The new pilot diagnostic searches integer FFT offsets ±32 and the four
scattered-pilot phases, interpolates the pilot channel estimate, and
extracts tentative differential TMCC soft bits / repeated 16-bit sync
candidates. It is deliberately **not** a BCH-verified TMCC decoder and
cannot currently output a playable TS from raw I/Q. A new uninterrupted
20ch recording is needed before implementing reliable frame continuity,
BCH validation and data FEC.


## Windows FC0013 crash in native async recorder (libusb -6 / access violation)

A Windows 11 DS-DT308SV reported native crashes with
`rtlsdr_demod_write_reg failed with -6`,
`Capture failed: exception: access violation writing 0x24` and a
second access violation in `BaseRtlSdr.__del__`. The previous
`read_samples_async()/cancel_read_async()` recorder has been
**disabled and removed from the capture path**. In upstream pyrtlsdr,
the async read/cancel error paths can call `self.close()`; a second
close while native USB transfers are unwinding is hazardous. This
is a credible failure mechanism, not a proven postmortem diagnosis
of the specific device crash. Do not retry the old async version.

The new `oneseg-capture` and GUI `Capture 3s for decoder…`
issue back-to-back **synchronous raw-byte USB reads**, copying the
reused ctypes buffer immediately. A separate Python thread performs
u8-to-complex64 conversion and file I/O using a bounded queue.
USB read, short read, writer lag, or write errors make the capture fail;
it does not report a successful sample-contiguous MPEG-TS recording.
The JSON sidecar sets `sample_continuity_verified: false`.

To recover from the access violation:

1. Close the old OneSeg process, SDR++ and other RTL-SDR programs.
2. Unplug DS-DT308SV and reconnect it after a short pause.
3. `git pull` then `uv sync`.
4. If `ch20_async.c64.partial` remains from the crashed run, inspect
   the filename and delete **only that incomplete partial file**;
   preserve the last successfully recorded .c64 and its .json.
5. With a newly named target, retry:

```powershell
uv run oneseg-doctor
uv run oneseg-capture --channel 20 --seconds 3 --gain 0 --ppm 0 --output ch20_safe.c64
uv run oneseg-quality ch20_safe.c64
uv run oneseg-lock ch20_safe.c64
uv run oneseg-pilots ch20_safe.c64 --seconds 1.2 --json ch20_safe_pilots.json
```

If `oneseg-doctor` itself crashes after unplug/replug or the new
sync-reader path still reports native access violations, do not
repeat the capture test: record the `pyrtlsdr` / bundled
`librtlsdr` versions, USB hardware ID and whether standalone
SDR++ still works; the native driver/DLL combination needs review.


## Shortened TMCC cyclic parity and layer diagnostics

The next diagnostic step after strong CP alignment and pilot coherence is to
verify the **82 parity bits** protecting ISDB-T TMCC B20..B121. Per the
Japanese digital terrestrial television transmission rules, TMCC uses a
shortened (184,102) difference-set cyclic code derived from (273,191).
The exact degree-82 generator polynomial is recorded in
[Japan's transmission rules, Annex 12 paragraph 2](https://laws.e-gov.go.jp/law/423M60000008087).
This is NOT the 188-byte MPEG-TS Reed-Solomon code and does not use the
15-bit energy dispersal PRBS. The code computes the binary polynomial
remainder of B20..B203 and accepts only zero-syndrome frames.

\`oneseg-pilots\` now attempts to align candidate 16-bit TMCC sync words
(B1..B16) with a **full 204-symbol frame**, verifies B20..B203 and only
then parses the partial-reception flag and layers A/B/C (modulation,
convolutional coding rate, Mode-3 interleaving and segment count).
Results are printed as the count of \`cyclic-parity-verified frames\`,
and accepted records are saved under \`bch_parity_verified_frames\` in
the JSON report. No bit errors are corrected. A nonzero count is
evidence of a protected TMCC frame, **not evidence that MPEG-TS or
live television is already supported**.

The prior ch20_safe.c64 CLI output (0 dB RF gain, repeated CP ~0.98,
pilot +12 bins / 0.974 coherence, six single / two 204-spaced sync
candidates) is promising but is not, by itself, a parity-verified TMCC
frame. With the matching sidecar still beside the capture, run:

\`\`\`powershell
git pull
uv sync
uv run oneseg-pilots ch20_safe.c64 --seconds 1.2 --json ch20_tmcc.json
\`\`\`

Inspect \`bch_parity_verified_frames\` in the output JSON. If none pass,
also try \`--seconds 3.0\` to search more symbols. In that case the
next step is detailed per-carrier differential-phase and frame timing
analysis using the *same* uploaded .c64/.json, not declaring a station
from sync candidates alone.


## ch20_safe recording: six consecutive TMCC parity-verified frames

The uploaded I/Q recordings with similar names are **different acquisitions**:

- First capture, UTC 2026-09-25 12:16:14, \`ch20_safe.c64\`:
  exactly **6,144,000** complex samples,
  3 full-scale component hits, approximate complex RMS 0.263,
  pilot offset +12 bins and fractional CFO around +113 Hz.
- Second capture, UTC 2026-09-25 12:35:07, uploaded as
  \`ch20_safe(1).c64\` with metadata filename
  \`ch20_safe.c64(1).json\`: also 6,144,000 samples but **695,587**
  full-scale I/Q hits (**11.32%**), RMS approximately 0.711,
  pilot offset +11 bins and fractional CFO around +428 Hz.

The supplied \`ch20_tmcc.json\` (capture field originally
\`ch20_safe.c64\`) matches the **second capture's RF offsets and CP
correlation**, not the first capture's. Independently running
the same processing stages on the second uploaded .c64 reproduced
2644 FFT symbols, 13 sync-word candidates and **six consecutively
parity-verified TMCC frames**, starting at differential-bit
positions 67, 271, 475, 679, 883 and 1087. All six frames share
the same protected information and parity; the alternating sync words
have zero errors. This confirms one central ISDB-T 1seg segment's
transmission metadata, **not** successful QPSK payload/FEC/TS decoding.

Reported TMCC: \`partial_reception_flag=true\`;
layer A: **QPSK 2/3**, Mode-3 time-interleave parameter **4**,
**1 segment**. Layer B: 64QAM 3/4, Mode-3 time interleaving
parameter 2, 12 segments; layer C unused. These parameters
are specific to the **recorded broadcast at that time**, not
universal settings. Because second recording clips 11.32% of
I/Q components, use the first 0-dB recording or reduce front-end
gain/antenna input if subsequent payload FEC fails. Fixed manual
gain 0 dB does not guarantee an unclipped ADC.

### Unmapped central Layer-A payload carriers

\`oneseg-pilots\` now excludes all 36 scattered pilots (symbol
dependent), four TMCC carriers and eight AC1 carriers per Mode-3
central segment. The first **432** carriers thus contain **384
complex-valued payload carriers** for each synchronized OFDM symbol;
the 433rd segment-edge carrier is excluded. See ARIB STD-B31
mode-3 center segment carrier layout and AC1/TMCC locations. No
frequency/time/bit deinterleaving or inner Viterbi is applied yet.

\`\`\`powershell
git pull
uv sync
uv run oneseg-pilots ch20_safe.c64 --seconds 3.0 --json ch20_layer_a.json --layer-a-output ch20_layer_a.npz
\`\`\`

The NPZ file includes \`equalized_payload_carriers\` of shape
\`(number_of_symbols, 384)\`, and
\`verified_tmcc_frame_start_fft_rows\`, which reference
only complete frames with passing TMCC cyclic parity.
The CLI also reports I/Q full-scale fraction and warns above
5%. This NPZ is **not a television/video file or a valid .ts**
and does not yet support live video.

# Verification and remaining acceptance

## Reproduce software checks

Use Python 3.12 and uv (no USB, antenna, audio device, or driver changes required):

```sh
uv sync --locked
uv run --locked ruff check .
QT_QPA_PLATFORM=offscreen uv run --locked pytest -q
uv run --locked oneseg --help
```

On PowerShell set `$env:QT_QPA_PLATFORM = 'offscreen'` before `uv run --locked pytest -q`.
The committed lockfile fixes the dependency set. Correctness lint deliberately
checks runtime/syntax errors, rather than imposing unrelated formatting changes.
CI runs the same tests on Windows and Linux. Hardware acceptance below is separate.

## Regression coverage

- Disk/serialization failure with a full telemetry queue cannot deadlock Stop.
- Consecutive raw windows preserve every returned USB sample using a shared
  remainder buffer. This asserts software byte accounting, not hardware timing.
- Existing capture metadata is rejected before opening USB; negative-gain
  commands and contradictory driver getters are retained separately.
- Strict TS conversion is transactional and never overwrites its input or a
  concurrently created output.
- Incremental transport discovery covers multi-packet PSI, CRC, pointer fields,
  continuity loss, duplicate packets, adaptation fields, table versions and
  removed services. A known external gap resets incomplete assemblies.
- Completed PAT and PMT can arrive in separate live windows; startup retains
  bounded authentic TS until the service is discoverable.
- Offline Qt controls exercise start/stop/close and file/live playback cancellation
  without a tuner. Actual sound-device behavior is not established by these tests.
- An independently forward-encoded synthetic fixture verifies carrier/bit/byte
  interleaving, Viterbi, RS correction and exactly 204-row boundaries at I=0/1/2/4.
- A generated 2.048 MS/s waveform with 73 Hz CFO traverses the complete
  I/Q → OFDM/pilots/TMCC → FEC → TS path without mocking the decoder.
  Recovered packets must match the original input byte-for-byte. This verifies
  a synthetic software channel, not an independent real broadcast reference.

## Real-device acceptance still required

Use the intended Windows 11 DS-DT308SV / RTL2832U + FC0013 setup with an appropriate
antenna and a locally active physical channel. Existing WinUSB setup must be
verified by the owner; software tests do not authorize driver changes.

1. Confirm tuner open/close and repeated start/stop/tune, with other tuner software
   closed. Preserve requested, commanded and read-back gain separately.
2. Capture a known-good, uninterrupted reference IQ signal with matching metadata.
   Confirm mode/guard, pilot alignment and protected TMCC parameters.
3. Compare independently known-good TS or a reference decoder against recovered
   TS, including continuity, PAT/PMT versions, PES boundaries, video and audio.
4. Measure long-running acquisition/DSP timings, RF loss/reacquisition and backpressure.
   No absence of overflow messages establishes hardware sample continuity.
5. Verify sustained real video/audio, service switching and graceful shutdown on
   the actual Windows sound/USB stack.

The current DSP still processes independent windows, supports a restricted
transmission mode, and may produce only partial TS. Passing software regressions
must not be described as confirmed gapless television or completion of all
hardware acceptance criteria in issue #1. No broadcast data is uploaded by tests.

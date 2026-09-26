import time

import pytest
from oneseg.live import LiveTsBuffer, ExperimentalLiveReceiver


def test_stream_blocks_and_wakes_when_data_arrives():
    from threading import Event, Thread
    b = LiveTsBuffer()
    got = []
    thread = Thread(target=lambda: got.append(b.read(188)))
    thread.start()
    assert thread.is_alive()
    b.push(b"\x47" + b"\0"*187)
    thread.join(timeout=2)
    assert got == [b"\x47" + b"\0"*187]
    b.close()


def test_stopping_stream_wakes_blocking_reader():
    from threading import Thread
    b = LiveTsBuffer()
    out=[]
    t=Thread(target=lambda: out.append(b.read(188)))
    t.start()
    b.close()
    t.join(timeout=2)
    assert out == [b""]


def test_backpressure_and_packet_alignment_fail_closed():
    b = LiveTsBuffer(limit_bytes=376)
    with pytest.raises(ValueError):
        b.push(b"bad")
    b.push(b"\x47"+b"\0"*187)
    b.push(b"\x47"+b"\1"*187)
    with pytest.raises(BufferError):
        b.push(b"\x47"+b"\2"*187)
    assert len(b.read(30)) == 30
    assert len(b.read(346)) == 346
    b.close()
    with pytest.raises(ValueError):
        b.push(b"\x47"+b"\3"*187)


def test_invalid_live_tuner_settings_rejected():
    with pytest.raises(ValueError):
        ExperimentalLiveReceiver(frequency_hz=515142857, ppm=0, gain=70)
    with pytest.raises(ValueError):
        ExperimentalLiveReceiver(frequency_hz=515142857, ppm=0, gain=0,
                                 chunk_seconds=.1)



def test_live_thread_pipeline_with_fake_usb_and_real_decoding_boundary(
    monkeypatch,
):
    """One full capture gets decoded while a different owner reads USB.

    No native async calls, and no user file or fake stream content can
    be published to the TS output without the decoder's own result.
    """
    from threading import Event
    from pathlib import Path
    from oneseg.continuous import CaptureCancelled

    decoded = Event()
    created = []
    capture_count = [0]

    class Device:
        def __init__(self):
            self.closed = False
            self.sample_rate = None
            self.center_freq = None
            self.gain = None
            self.freq_correction = 0

        def close(self):
            self.closed = True

        def read_samples_async(self, *a, **k):
            raise AssertionError("unsafe native async MUST NOT be used")

    fake = Device()

    def fake_record(device, path, *, samples_required, metadata, cancelled):
        assert device is fake
        assert samples_required == 2048000
        capture_count[0] += 1
        if capture_count[0] > 1:
            assert decoded.wait(timeout=3)
            raise CaptureCancelled("test ended after one window")
        path.write_bytes(b"\0" * 8)
        path.with_suffix(".c64.json").write_text("{}")
        created.append(path)

    def fake_decode(source, target, *, seconds, max_ofdm_symbols):
        assert source.is_file()
        assert source.with_suffix(".c64.json").is_file()
        assert seconds == 1 and max_ofdm_symbols == 3000
        target.write_bytes(b"\x47" + b"\0" * 187)
        target.with_suffix(".ts.json").write_text("{}")
        decoded.set()
        return {
            "rs_and_ts_accepted_packets": 1,
            "rejected_rs_or_invalid_ts_packets": 0,
            "pat_programs": {},
            "input_overload_warning": False,
        }

    monkeypatch.setattr("oneseg.live.record_stream", fake_record)
    worker = ExperimentalLiveReceiver(
        frequency_hz=515142857,
        ppm=0,
        gain=-3.0,
        chunk_seconds=1,
        device_factory=lambda: fake,
        decode_function=fake_decode,
    )
    worker.run()
    assert decoded.is_set()
    assert fake.closed
    assert fake.center_freq == 515142857
    assert fake.gain == -3.0
    assert capture_count[0] == 2
    assert not created[0].exists()  # temp I/Q deleted only after decode join

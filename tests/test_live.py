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

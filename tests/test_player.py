"""Real PyAV/Qt playback plus deterministic cancellation/resource regressions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fractions import Fraction
from io import BytesIO
from pathlib import Path
import time

import av
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from oneseg.player import TransportPlayer


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


def video_frame(pts=0):
    frame = av.VideoFrame.from_ndarray(np.full((32, 48, 3), 80, dtype=np.uint8), format="rgb24")
    frame.pts = pts
    frame.time_base = Fraction(1, 25)
    return frame


class FakeContainer:
    def __init__(self, frames):
        self.frames = frames
        self.closed = False

    def decode(self):
        yield from self.frames

    def close(self):
        self.closed = True


def test_player_construct_without_media_or_audio_device():
    player = TransportPlayer(Path("synthetic.ts"))
    assert player.path.name == "synthetic.ts"
    assert not player.stop_event.is_set()
    player.stop()
    assert player.stop_event.is_set()


def test_pre_cancelled_player_does_not_open_input(monkeypatch):
    monkeypatch.setattr(av, "open", lambda *a, **k: pytest.fail("opened cancelled player"))
    player = TransportPlayer(Path("synthetic.ts"))
    player.stop()
    player.run()


def test_unnamed_binary_input_is_playable_and_image_owns_pixels(monkeypatch):
    source = BytesIO(b"synthetic")
    frame = video_frame()
    container = FakeContainer([frame])
    monkeypatch.setattr(av, "open", lambda *a, **k: container)
    player = TransportPlayer(source)
    images, failures, statuses = [], [], []
    player.image_ready.connect(images.append)
    player.failed.connect(failures.append)
    player.status.connect(statuses.append)
    player.run()
    assert failures == []
    assert len(images) == 1
    assert images[0].width() == 48
    assert images[0].pixelColor(0, 0).red() == 80
    assert any("MPEG-TS stream" in status for status in statuses)
    assert container.closed


def test_stop_while_opening_live_input_is_not_a_media_failure(monkeypatch):
    player = TransportPlayer(BytesIO())
    def interrupted_open(*args, **kwargs):
        player.stop()
        raise EOFError("input closed while probing")
    monkeypatch.setattr(av, "open", interrupted_open)
    failures = []
    player.failed.connect(failures.append)
    player.run()
    assert failures == []


def test_invalid_file_reports_failure(app, tmp_path):
    invalid = tmp_path / "invalid.ts"
    invalid.write_bytes(b"not an MPEG transport stream")
    player = TransportPlayer(invalid)
    failures = []
    player.failed.connect(failures.append)
    player.start()
    wait_until(app, lambda: not player.isRunning())
    assert len(failures) == 1
    assert any(kind in failures[0] for kind in ("InvalidDataError", "EOFError"))


def test_audio_device_failure_only_attempted_once(monkeypatch):
    frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
    frame.sample_rate = 48000
    container = FakeContainer([frame, frame, frame])
    monkeypatch.setattr(av, "open", lambda *a, **k: container)
    attempts = []
    def unavailable():
        attempts.append(True)
        raise RuntimeError("no audio hardware")
    monkeypatch.setattr("oneseg.wfm.AudioQueue", unavailable)
    player = TransportPlayer(Path("audio.ts"))
    failures, statuses = [], []
    player.failed.connect(failures.append)
    player.status.connect(statuses.append)
    player.run()
    assert attempts == [True]
    assert sum("No audio output" in status for status in statuses) == 1
    assert failures == []
    assert container.closed


def test_audio_close_failure_still_closes_demuxer(monkeypatch):
    frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
    frame.sample_rate = 48000
    container = FakeContainer([frame])
    monkeypatch.setattr(av, "open", lambda *a, **k: container)
    class AudioSink:
        def feed(self, audio):
            pass
        def close(self):
            raise RuntimeError("device vanished")
    monkeypatch.setattr("oneseg.wfm.AudioQueue", AudioSink)
    player = TransportPlayer(Path("audio.ts"))
    failures = []
    player.failed.connect(failures.append)
    player.run()
    assert container.closed
    assert any("Could not close audio output" in failure for failure in failures)


def test_stop_interrupts_timestamp_wait_without_emitting_late_frame(app, monkeypatch):
    container = FakeContainer([video_frame(0), video_frame(250)])
    monkeypatch.setattr(av, "open", lambda *a, **k: container)
    player = TransportPlayer(Path("slow.ts"))
    images = []
    player.image_ready.connect(images.append)
    player.start()
    try:
        wait_until(app, lambda: len(images) == 1)
        player.stop()
        assert player.wait(500)
        app.processEvents()
        assert len(images) == 1
        assert container.closed
    finally:
        player.stop()
        player.wait(1000)


def test_probe_time_does_not_consume_playback_clock(monkeypatch):
    container = FakeContainer([video_frame(0), video_frame(25)])
    clock = [0.0]
    def slow_open(*args, **kwargs):
        clock[0] = 10.0
        return container
    monkeypatch.setattr(av, "open", slow_open)
    monkeypatch.setattr("oneseg.player.time.monotonic", lambda: clock[0])
    player = TransportPlayer(Path("slow-probe.ts"))
    waits = []
    def record_wait(delay):
        waits.append(delay)
        clock[0] += delay
        return False
    monkeypatch.setattr(player.stop_event, "wait", record_wait)
    player.run()
    assert waits == [1.0]


def test_real_mpegts_decodes_on_qthread_without_audio_device(app, tmp_path, monkeypatch):
    path = tmp_path / "actual.ts"
    with av.open(str(path), "w", format="mpegts") as output:
        stream = output.add_stream("mpeg2video", rate=25)
        stream.width, stream.height = 48, 32
        stream.pix_fmt = "yuv420p"
        for index in range(4):
            for packet in stream.encode(video_frame(index)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    monkeypatch.setattr("oneseg.wfm.AudioQueue", lambda: pytest.fail("video-only file opened audio"))
    player = TransportPlayer(path)
    images, failures = [], []
    player.image_ready.connect(images.append)
    player.failed.connect(failures.append)
    player.start()
    try:
        wait_until(app, lambda: not player.isRunning())
        assert failures == []
        assert len(images) == 4
        assert all(image.width() == 48 and image.height() == 32 for image in images)
    finally:
        player.stop()
        player.wait(1000)


def test_stop_during_open_does_not_enter_decode(monkeypatch):
    player = TransportPlayer(Path("cancelled-open.ts"))
    container = FakeContainer([])
    monkeypatch.setattr(container, "decode", lambda: pytest.fail("decode entered after Stop"))
    def cancelled_open(*args, **kwargs):
        player.stop()
        return container
    monkeypatch.setattr(av, "open", cancelled_open)
    player.run()
    assert container.closed

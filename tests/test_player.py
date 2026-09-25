from pathlib import Path

from oneseg.player import TransportPlayer


def test_player_construct_without_media_or_audio_device():
    player = TransportPlayer(Path("synthetic.ts"))
    assert player.path.name == "synthetic.ts"
    assert not player.stop_event.is_set()
    player.stop()
    assert player.stop_event.is_set()

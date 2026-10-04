"""Atomic, no-clobber publication of a completed file on its own filesystem."""
from __future__ import annotations

import os
from pathlib import Path


def publish_new_file(source: Path, target: Path) -> None:
    """Publish without replacing a concurrent output.

    Windows rename is no-clobber and also supports FAT/exFAT recording drives.
    POSIX rename overwrites, so use a same-filesystem hard link instead. Callers
    own source cleanup and should unlink(missing_ok=True) after publication.
    Never substitute os.replace or a check-then-rename sequence on POSIX.
    """
    if os.name == "nt":
        os.rename(source, target)
    else:
        os.link(source, target)

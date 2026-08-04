"""Console helpers shared by command-line entry points."""

from __future__ import annotations

import sys
from typing import TextIO


def _reconfigure_utf8(stream: TextIO | None) -> None:
    """Make one Python text stream Unicode-safe when it supports reconfigure()."""
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    except (OSError, ValueError):
        # Detached/closed streams can appear in test runners and embedding hosts.
        # Console configuration is best effort; experiment data is persisted as
        # UTF-8 independently of these display streams.
        return


def configure_utf8_stdio() -> None:
    """Configure stdout and stderr so arbitrary model text cannot crash a run."""
    _reconfigure_utf8(sys.stdout)
    _reconfigure_utf8(sys.stderr)

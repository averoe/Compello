"""Shared rendering style for Compello's terminal output.

Provides a small glyph set (emoji/unicode by default, ASCII fallback for pipes,
log files, or terminals with a non-UTF-8 encoding) and box-drawing helpers so
the pre-flight shield, runtime insight blocks, and post-training report all look
consistent. Auto-detects whether stdout can encode unicode and degrades
gracefully instead of raising ``UnicodeEncodeError`` in a training log.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

WIDTH = 80

_UNICODE = {
    "green": "\U0001F7E2",   # green circle
    "yellow": "\U0001F7E1",  # yellow circle
    "red": "\U0001F534",     # red circle
    "gear": "\u2699",        # gear
    "warn": "\u26A0",        # warning
    "check": "\u2705",       # check mark
    "cross": "\u274C",       # cross mark
    "bulb": "\U0001F4A1",    # light bulb
    "wrench": "\U0001F527",  # wrench
    "mag": "\U0001F50D",     # magnifier
    "chart": "\U0001F4CA",   # bar chart
    "arrow": "->",
}

_ASCII = {
    "green": "[OK]",
    "yellow": "[!]",
    "red": "[X]",
    "gear": "*",
    "warn": "[!]",
    "check": "[OK]",
    "cross": "[X]",
    "bulb": "[i]",
    "wrench": "[fix]",
    "mag": "[?]",
    "chart": "[#]",
    "arrow": "->",
}


def _stdout_supports_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    if "utf" in enc:
        return True
    # try encoding a sample glyph
    try:
        "\U0001F7E2".encode(enc or "ascii")
        return True
    except Exception:
        return False


@dataclass
class Style:
    unicode: bool = True
    width: int = WIDTH

    @classmethod
    def auto(cls) -> "Style":
        return cls(unicode=_stdout_supports_unicode())

    def g(self, key: str) -> str:
        table = _UNICODE if self.unicode else _ASCII
        return table.get(key, "")

    # -- box drawing (matches the '=' banner style in the spec mockups) --
    def rule(self, char: str = "=") -> str:
        return char * self.width

    def banner(self, title: str, char: str = "=") -> str:
        line = self.rule(char)
        return f"{line}\n{title.center(self.width)}\n{line}"

    def section(self, char: str = "-") -> str:
        return self.rule(char)

    def status_glyph(self, compliant: int, total: int) -> str:
        if total == 0 or compliant == total:
            return self.g("green")
        if compliant == 0:
            return self.g("red")
        return self.g("yellow")

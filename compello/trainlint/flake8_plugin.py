"""Flake8 plugin entry point for trainlint (Section 8.4).

Flake8 discovers a plugin via the ``flake8.extension`` entry point (declared in
pyproject.toml). The plugin class is instantiated with the parsed AST and the
filename, and ``run()`` yields ``(line, col, message, type)`` tuples. Codes are
prefixed ``TL`` (TrainLint) so they are namespaced away from other checkers.

This reuses the exact same rule engine as the CLI -- no rule logic is
duplicated here; it only adapts the output shape flake8 expects.
"""

from __future__ import annotations

import ast
from typing import Any, Iterator, Tuple

from .core import ERROR, lint_source

_CODE_PREFIX = "TL"


class TrainlintFlake8Plugin:
    name = "trainlint"
    version = "0.1.0"

    def __init__(self, tree: ast.AST, filename: str = "(none)"):
        self._tree = tree
        self._filename = filename

    def run(self) -> Iterator[Tuple[int, int, str, Any]]:
        try:
            source = ast.unparse(self._tree)
        except Exception:
            return
        for issue in lint_source(source, filename=self._filename):
            code = f"{_CODE_PREFIX}{'0' if issue.severity == ERROR else '9'}01"
            msg = f"{code} {issue.rule}: {issue.message}"
            yield issue.line, issue.col, msg, type(self)

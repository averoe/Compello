"""Trainlint -- the companion static linter (Section 8)."""

from .core import (
    ERROR,
    WARNING,
    LintIssue,
    detect_backends,
    lint_file,
    lint_source,
)

__all__ = [
    "lint_source",
    "lint_file",
    "detect_backends",
    "LintIssue",
    "ERROR",
    "WARNING",
]

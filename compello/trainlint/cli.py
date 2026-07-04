"""Trainlint CLI (Section 8.4).

Usage:
    trainlint path/to/script.py [more.py ...]
    trainlint --backend pytorch,jax path/to/script.py

Auto-detects active backends by import scanning unless --backend is given.
Exit code is non-zero if any error-severity issue is found.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from .core import ERROR, LintIssue, lint_file


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="trainlint", description=__doc__)
    parser.add_argument("paths", nargs="+", help="Python files to lint")
    parser.add_argument(
        "--backend",
        default=None,
        help="comma-separated backends to force (pytorch,tensorflow,jax); "
             "default: auto-detect via imports",
    )
    parser.add_argument(
        "--shield",
        action="store_true",
        help="render the rich Pre-Flight Static Shield (why + actionable fix) "
             "instead of terse one-line diagnostics",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="force ASCII glyphs instead of emoji/unicode",
    )
    args = parser.parse_args(argv)

    backends: Optional[List[str]] = (
        [b.strip() for b in args.backend.split(",")] if args.backend else None
    )

    total_errors = 0
    for path in args.paths:
        try:
            issues = lint_file(path, backends=backends)
        except SyntaxError as exc:
            print(f"{path}: syntax error: {exc}", file=sys.stderr)
            total_errors += 1
            continue

        if args.shield:
            from ..preflight_render import render_preflight_shield
            from ..report_style import Style

            style = Style(unicode=not args.ascii)
            print(render_preflight_shield(issues, script=path, style=style))
            total_errors += sum(1 for i in issues if i.severity == ERROR)
        else:
            for issue in issues:
                print(issue.format(path))
                if issue.severity == ERROR:
                    total_errors += 1

    if total_errors:
        if not args.shield:
            print(f"\ntrainlint: {total_errors} error(s) found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

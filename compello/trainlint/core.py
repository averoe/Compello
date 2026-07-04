"""Trainlint core: AST infrastructure and backend detection (Section 8).

Trainlint depends only on the standard-library ``ast`` module. It never imports
or executes user model code, so it cannot be broken by a version mismatch or a
missing GPU/CUDA/TPU install. Backend-specific rule sets are selected by
static import-scanning of the target file, not by importing the frameworks.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

# severities
ERROR = "error"
WARNING = "warning"


@dataclass
class LintIssue:
    rule: str
    severity: str
    message: str
    line: int
    col: int
    backend: str = "generic"

    def format(self, filename: str = "<source>") -> str:
        return f"{filename}:{self.line}:{self.col}: [{self.severity}] {self.rule}: {self.message}"


class Rule:
    """Base class for a lint rule. Subclasses implement ``check(tree)``."""

    name = "rule"
    backend = "generic"

    def check(self, tree: ast.AST, source: str) -> List[LintIssue]:  # pragma: no cover
        raise NotImplementedError


def detect_backends(tree: ast.AST) -> Set[str]:
    """Return the set of frameworks imported in ``tree`` (Section 8.4)."""
    backends: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _classify(alias.name, backends)
        elif isinstance(node, ast.ImportFrom) and node.module:
            _classify(node.module, backends)
    return backends


def _classify(module: str, backends: Set[str]) -> None:
    root = module.split(".")[0]
    if root == "torch":
        backends.add("pytorch")
    elif root in ("tensorflow", "keras"):
        backends.add("tensorflow")
    elif root in ("jax", "flax", "optax"):
        backends.add("jax")


def lint_source(
    source: str,
    *,
    filename: str = "<source>",
    backends: Optional[Sequence[str]] = None,
) -> List[LintIssue]:
    """Lint Python ``source`` and return issues.

    If ``backends`` is None the active backends are auto-detected by import
    scanning; the relevant per-backend rule sets plus the always-on
    Compello/generic rules are applied.
    """
    from . import jax_rules, pytorch_rules, tensorflow_rules

    tree = ast.parse(source, filename=filename)
    active = set(backends) if backends is not None else detect_backends(tree)

    rules: List[Rule] = list(pytorch_rules.COMPELLO_RULES)  # always-on
    if "pytorch" in active:
        rules.extend(pytorch_rules.RULES)
    if "tensorflow" in active:
        rules.extend(tensorflow_rules.RULES)
    if "jax" in active:
        rules.extend(jax_rules.RULES)

    issues: List[LintIssue] = []
    for rule in rules:
        issues.extend(rule.check(tree, source))
    issues.sort(key=lambda i: (i.line, i.col, i.rule))
    return issues


def lint_file(path: str, backends: Optional[Sequence[str]] = None) -> List[LintIssue]:
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    return lint_source(source, filename=path, backends=backends)


# --- shared AST helpers ---------------------------------------------------

def function_defs(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def call_name(node: ast.Call) -> str:
    """Return a dotted name for a call target, e.g. 'torch.compile' or 'x.append'."""
    return attr_chain(node.func)


def attr_chain(node: ast.AST) -> str:
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def has_decorator(fn: ast.AST, dotted: str) -> bool:
    for dec in getattr(fn, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if attr_chain(target).endswith(dotted):
            return True
    return False


def calls_in(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            yield n

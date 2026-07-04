"""TensorFlow / Keras trainlint rules (Section 8.3, new in v4)."""

from __future__ import annotations

import ast
from typing import List

from .core import (
    ERROR,
    WARNING,
    LintIssue,
    Rule,
    attr_chain,
    call_name,
    calls_in,
    function_defs,
    has_decorator,
)

_NON_DIFF_OPS = ("argmax", "argmin", "round", "floor", "ceil", "sign")


class UntrackedVariableRule(Rule):
    """Controller state assigned as a plain Python attr inside @tf.function
    gets baked as a constant by AutoGraph (Section 4.6b / 8.3)."""

    name = "untracked-variable"
    backend = "tensorflow"
    _STATE_HINTS = ("integral", "ema", "smoothed", "buffer", "accumulator", "counter", "baseline")

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for fn in function_defs(tree):
            if not has_decorator(fn, "function"):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.AugAssign):
                    tname = attr_chain(node.target)
                    if self._is_state(tname):
                        issues.append(self._issue(tname, node.lineno, node.col_offset))
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        tname = attr_chain(t)
                        if self._is_state(tname) and not _is_tf_variable(node.value):
                            issues.append(self._issue(tname, node.lineno, node.col_offset))
        return issues

    def _is_state(self, name: str) -> bool:
        low = name.lower()
        return name.startswith("self.") and any(h in low for h in self._STATE_HINTS)

    def _issue(self, tname, line, col) -> LintIssue:
        return LintIssue(
            self.name, ERROR,
            f"controller state '{tname}' mutated as a plain Python attribute "
            "inside @tf.function -- AutoGraph bakes it as a constant. Use "
            "tf.Variable(..., trainable=False) with .assign_add() (4.6b).",
            line, col, self.backend,
        )


class MissingTapeContextRule(Rule):
    name = "missing-tape-context"
    backend = "tensorflow"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for fn in function_defs(tree):
            grad_calls = [c for c in calls_in(fn) if call_name(c).endswith(".gradient")]
            if not grad_calls:
                continue
            has_tape = any(
                isinstance(n, ast.With) and any(
                    attr_chain(it.context_expr.func if isinstance(it.context_expr, ast.Call)
                               else it.context_expr).endswith("GradientTape")
                    for it in n.items
                )
                for n in ast.walk(fn)
            )
            if not has_tape:
                c = grad_calls[0]
                issues.append(LintIssue(
                    self.name, ERROR,
                    "gradient computation without an active tf.GradientTape() "
                    "context.",
                    c.lineno, c.col_offset, self.backend,
                ))
        return issues


class NonDifferentiableAssertionRule(Rule):
    name = "nondiff-op-in-assertion"
    backend = "tensorflow"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and call_name(node).endswith("expect")):
                continue
            for inner in calls_in(node):
                cn = call_name(inner)
                if any(cn.endswith(op) for op in _NON_DIFF_OPS):
                    issues.append(LintIssue(
                        self.name, WARNING,
                        f"assertion target uses non-differentiable op '{cn}' "
                        "without a registered relaxation -- the penalty will "
                        "have no gradient.",
                        inner.lineno, inner.col_offset, self.backend,
                    ))
        return issues


class InGraphVariableCreationRule(Rule):
    name = "in-graph-variable-creation"
    backend = "tensorflow"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for fn in function_defs(tree):
            if not has_decorator(fn, "function"):
                continue
            for c in calls_in(fn):
                if call_name(c).endswith("Variable"):
                    issues.append(LintIssue(
                        self.name, WARNING,
                        "tf.Variable created inside a traced @tf.function -- "
                        "create it once at init; in-graph creation causes "
                        "retracing / memory growth.",
                        c.lineno, c.col_offset, self.backend,
                    ))
        return issues


def _is_tf_variable(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and call_name(node).endswith("Variable")


RULES = [
    UntrackedVariableRule(),
    MissingTapeContextRule(),
    NonDifferentiableAssertionRule(),
    InGraphVariableCreationRule(),
]

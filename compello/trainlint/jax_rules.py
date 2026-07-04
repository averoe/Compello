"""JAX trainlint rules (Section 8.3, new in v4)."""

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


def _is_jitted(fn: ast.AST) -> bool:
    return has_decorator(fn, "jit") or has_decorator(fn, "pmap")


class ImpureMutationRule(Rule):
    """Mutation inside a jitted function breaks functional purity (4.2/4.7b/10.1)."""

    name = "impure-mutation"
    backend = "jax"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for fn in function_defs(tree):
            if not _is_jitted(fn):
                continue
            for node in ast.walk(fn):
                # aug-assign to an attribute or subscript (in-place state mutation)
                if isinstance(node, ast.AugAssign) and isinstance(
                    node.target, (ast.Attribute, ast.Subscript)
                ):
                    issues.append(self._issue(attr_chain(_root(node.target)),
                                               node.lineno, node.col_offset))
                # item assignment: x[i] = ...
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Subscript):
                            issues.append(self._issue(attr_chain(t.value),
                                                       node.lineno, node.col_offset))
        return issues

    def _issue(self, name, line, col) -> LintIssue:
        return LintIssue(
            self.name, ERROR,
            f"in-place mutation of '{name}' inside a jitted function violates "
            "JAX functional purity -- thread state via an explicit PyTree "
            "return value instead (10.1).",
            line, col, self.backend,
        )


class UntracedSideEffectRule(Rule):
    name = "untraced-side-effect"
    backend = "jax"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for fn in function_defs(tree):
            if not _is_jitted(fn):
                continue
            for c in calls_in(fn):
                cn = call_name(c)
                if cn == "print" or cn.endswith((".info", ".debug", ".warning")):
                    issues.append(LintIssue(
                        self.name, WARNING,
                        f"'{cn}' inside a jitted function only runs at trace "
                        "time, not every call -- use jax.debug.print for "
                        "per-step output.",
                        c.lineno, c.col_offset, self.backend,
                    ))
        return issues


class DonationHazardRule(Rule):
    name = "donation-hazard"
    backend = "jax"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node).endswith("jit"):
                if any(k.arg == "donate_argnums" for k in node.keywords):
                    issues.append(LintIssue(
                        self.name, WARNING,
                        "jax.jit(donate_argnums=...) donates a buffer; if "
                        "Compello's controller-state threading reads it after "
                        "the call this risks a use-after-donation error (8.3).",
                        node.lineno, node.col_offset, self.backend,
                    ))
        return issues


class DistributedAutoMisconfigRule(Rule):
    """Flags a dict literal pairing jax_native backend with distributed: auto (4.1b)."""

    name = "distributed-auto-misconfiguration"
    backend = "jax"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            found = {}
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    found[k.value] = v.value
            if found.get("backend") == "jax_native" and found.get("distributed") == "auto":
                issues.append(LintIssue(
                    self.name, ERROR,
                    "distributed: auto cannot be resolved under backend: "
                    "jax_native -- provide an explicit axis_name (4.1b).",
                    node.lineno, node.col_offset, self.backend,
                ))
        return issues


def _root(node: ast.AST) -> ast.AST:
    cur = node
    while isinstance(cur, ast.Subscript):
        cur = cur.value
    return cur


RULES = [
    ImpureMutationRule(),
    UntracedSideEffectRule(),
    DonationHazardRule(),
    DistributedAutoMisconfigRule(),
]

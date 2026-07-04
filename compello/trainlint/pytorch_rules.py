"""PyTorch trainlint rules + always-on Compello rules (Section 8.3)."""

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


class DetachedLossRule(Rule):
    name = "detached-loss"
    backend = "pytorch"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for node in ast.walk(tree):
            # list.append(<loss-ish tensor>) without .item()/.detach()
            if isinstance(node, ast.Call) and call_name(node).endswith(".append"):
                if node.args and _is_bare_loss(node.args[0]):
                    issues.append(LintIssue(
                        self.name, WARNING,
                        "appending a tensor to a list for logging without "
                        ".item()/.detach() retains the autograd graph (memory leak).",
                        node.lineno, node.col_offset, self.backend,
                    ))
            # accumulation: total += loss  (without .item())
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                tname = attr_chain(node.target)
                if ("loss" in tname.lower() or "total" in tname.lower()) and _is_bare_loss(node.value):
                    issues.append(LintIssue(
                        self.name, WARNING,
                        f"accumulating into '{tname}' without .item()/.detach() "
                        "retains the graph across steps (memory leak).",
                        node.lineno, node.col_offset, self.backend,
                    ))
        return issues


class ZeroGradRule(Rule):
    name = "zero-grad"
    backend = "pytorch"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for fn in function_defs(tree):
            names = {call_name(c) for c in calls_in(fn)}
            has_backward = any(n.endswith(".backward") for n in names)
            has_zero = any(n.endswith(".zero_grad") for n in names)
            if has_backward and not has_zero:
                issues.append(LintIssue(
                    self.name, WARNING,
                    f"function '{fn.name}' calls .backward() but never "
                    ".zero_grad() -- gradients silently accumulate across steps.",
                    fn.lineno, fn.col_offset, self.backend,
                ))
        return issues


class TrainEvalModeRule(Rule):
    name = "train-eval-mode"
    backend = "pytorch"

    def check(self, tree, source) -> List[LintIssue]:
        names = {call_name(c) for c in calls_in(tree)}
        has_backward = any(n.endswith(".backward") for n in names)
        sets_mode = any(n.endswith(".train") or n.endswith(".eval") for n in names)
        if has_backward and not sets_mode:
            return [LintIssue(
                self.name, WARNING,
                "training loop present but model.train()/model.eval() is never "
                "called -- BatchNorm/Dropout may behave incorrectly during "
                "training vs. evaluation.",
                1, 0, self.backend,
            )]
        return []


class NoGradRule(Rule):
    name = "no-grad"
    backend = "pytorch"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for fn in function_defs(tree):
            lname = fn.name.lower()
            if not any(k in lname for k in ("eval", "val", "test", "infer", "predict")):
                continue
            uses_no_grad = _uses_no_grad(fn)
            calls_model = any(
                not n.endswith((".backward", ".step", ".zero_grad", ".item"))
                for n in {call_name(c) for c in calls_in(fn)}
            )
            if calls_model and not uses_no_grad:
                issues.append(LintIssue(
                    self.name, WARNING,
                    f"'{fn.name}' looks like eval/inference but has no "
                    "torch.no_grad()/@torch.inference_mode() -- wastes memory "
                    "building an unused autograd graph.",
                    fn.lineno, fn.col_offset, self.backend,
                ))
        return issues


class InplaceLeafRule(Rule):
    name = "inplace-on-leaf"
    backend = "pytorch"
    _INPLACE = {"add_", "mul_", "sub_", "div_", "clamp_", "copy_", "zero_"}

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in self._INPLACE:
                    base = attr_chain(node.func.value).lower()
                    if any(k in base for k in ("param", "weight", "leaf")):
                        issues.append(LintIssue(
                            self.name, WARNING,
                            f"in-place op '.{node.func.attr}' on '{attr_chain(node.func.value)}' "
                            "which looks like a leaf tensor requiring grad -- breaks autograd.",
                            node.lineno, node.col_offset, self.backend,
                        ))
        return issues


class DataLoaderShuffleRule(Rule):
    name = "dataloader-shuffle"
    backend = "pytorch"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            if not call_name(node.value).endswith("DataLoader"):
                continue
            target = attr_chain(node.targets[0]).lower() if node.targets else ""
            shuffle = _kw_bool(node.value, "shuffle")
            if shuffle is None:
                continue
            is_eval = any(k in target for k in ("val", "test", "eval"))
            is_train = "train" in target
            if is_eval and shuffle is True:
                issues.append(LintIssue(
                    self.name, WARNING,
                    f"validation/test loader '{target}' has shuffle=True.",
                    node.value.lineno, node.value.col_offset, self.backend,
                ))
            elif is_train and shuffle is False:
                issues.append(LintIssue(
                    self.name, WARNING,
                    f"training loader '{target}' has shuffle=False.",
                    node.value.lineno, node.value.col_offset, self.backend,
                ))
        return issues


class CompiledOptimizerStepRule(Rule):
    name = "compiled-optimizer-step"
    backend = "pytorch"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        # gather function names that contain both backward and optimizer.step
        risky_fns = set()
        for fn in function_defs(tree):
            names = {call_name(c) for c in calls_in(fn)}
            if any(n.endswith(".backward") for n in names) and any(
                n.endswith(".step") for n in names
            ):
                risky_fns.add(fn.name)
            # decorator form: @torch.compile on such a function
            if has_decorator(fn, "compile") and fn.name in risky_fns:
                issues.append(self._issue(fn.lineno, fn.col_offset))
        # call form: torch.compile(train_step)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and call_name(node).endswith("compile"):
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in risky_fns:
                        issues.append(self._issue(node.lineno, node.col_offset))
        return issues

    def _issue(self, line, col) -> LintIssue:
        return LintIssue(
            self.name, ERROR,
            "optimizer step appears compiled inside the same region as "
            "backward; a gradient_surgery constraint cannot run in the required "
            "post-backward/pre-step gap (CompiledOptimizerStepConflictError, 4.6). "
            "Split the compiled region and leave optimizer.step() eager.",
            line, col, self.backend,
        )


class AmbiguousAssertionRule(Rule):
    """Always-on Compello rule (Section 3.1a / 8.3)."""

    name = "ambiguous-assertion"
    backend = "compello"

    def check(self, tree, source) -> List[LintIssue]:
        issues: List[LintIssue] = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and call_name(node).endswith("expect")):
                continue
            if not node.args:
                continue
            target = node.args[0]
            has_type = any(k.arg == "assertion_type" for k in node.keywords)
            # keyword assertions are unambiguous
            kw_assertion = any(
                k.arg in {"invariant_to", "parity_across", "lipschitz_bound",
                          "monotonic_in", "consistent_across"}
                for k in node.keywords
            )
            if has_type or kw_assertion:
                continue
            if not _is_wrapped_target(target):
                issues.append(LintIssue(
                    self.name, ERROR,
                    "expect() target is not compello.wrap(...)'d and no "
                    "assertion_type= was given -- dispatch is ambiguous "
                    "(AmbiguousAssertionError, 3.1a).",
                    node.lineno, node.col_offset, self.backend,
                ))
        return issues


# --- helpers --------------------------------------------------------------

def _is_bare_loss(node: ast.AST) -> bool:
    """True if ``node`` is a loss-like tensor expression not detached."""
    if isinstance(node, ast.Call):
        n = call_name(node)
        if n.endswith((".item", ".detach", ".cpu", ".tolist")) or n in ("float", "int"):
            return False
    if isinstance(node, ast.Name):
        return "loss" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return "loss" in node.attr.lower()
    return False


def _uses_no_grad(fn: ast.AST) -> bool:
    if has_decorator(fn, "inference_mode") or has_decorator(fn, "no_grad"):
        return True
    for node in ast.walk(fn):
        if isinstance(node, ast.With):
            for item in node.items:
                ce = item.context_expr
                target = ce.func if isinstance(ce, ast.Call) else ce
                if attr_chain(target).endswith(("no_grad", "inference_mode")):
                    return True
    return False


def _kw_bool(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name and isinstance(k.value, ast.Constant):
            return k.value.value
    return None


def _is_wrapped_target(node: ast.AST) -> bool:
    # compello.wrap(x)  OR  compello.wrap(x)[i]  OR model.output-style attribute
    if isinstance(node, ast.Subscript):
        return _is_wrapped_target(node.value)
    if isinstance(node, ast.Call):
        return call_name(node).endswith("wrap")
    if isinstance(node, ast.Attribute):
        # allow wrapped_model.output
        return node.attr == "output"
    return False


COMPELLO_RULES = [AmbiguousAssertionRule()]
RULES = [
    DetachedLossRule(),
    ZeroGradRule(),
    TrainEvalModeRule(),
    NoGradRule(),
    InplaceLeafRule(),
    DataLoaderShuffleRule(),
    CompiledOptimizerStepRule(),
]

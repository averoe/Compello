"""Pre-Flight Static Shield renderer (Section 8).

Turns trainlint issues and preflight errors into the boxed, plain-English shield
output: for each blocking problem, WHY it will hurt the run and an ACTIONABLE
FIX (with a before/after template where one exists). This is the terminal state
a user hits before any GPU memory is allocated.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .fixtemplates import get_fix_template
from .report_style import Style
from .trainlint.core import ERROR, LintIssue


def render_preflight_shield(
    issues: Optional[Sequence[LintIssue]] = None,
    *,
    preflight_errors: Optional[Sequence[str]] = None,
    script: str = "<script>",
    style: Optional[Style] = None,
) -> str:
    style = style or Style.auto()
    issues = list(issues or [])
    preflight_errors = list(preflight_errors or [])

    out: List[str] = [style.banner("COMPELLO PRE-FLIGHT STATIC SHIELD")]
    out.append(f"[trainlint] Scanning {script!r} using Python AST...")
    out.append("")

    errors = [i for i in issues if i.severity == ERROR]
    warnings = [i for i in issues if i.severity != ERROR]

    if not errors and not preflight_errors:
        out.append(f"{style.g('check')} ALL PRE-FLIGHT CHECKS PASSED")
        if warnings:
            out.append("")
            out.append(f"{style.g('warn')} {len(warnings)} non-blocking warning(s):")
            for w in warnings:
                out.append(f"   {style.g('arrow')} {w.rule} ({w.backend}) L{w.line}: {w.message}")
        out.append(style.rule())
        return "\n".join(out)

    for issue in errors:
        out.extend(_render_issue(issue, script, style))
        out.append("")

    for msg in preflight_errors:
        out.extend(_render_preflight_error(msg, style))
        out.append("")

    if warnings:
        out.append(f"{style.g('warn')} {len(warnings)} non-blocking warning(s):")
        for w in warnings:
            out.append(f"   {style.g('arrow')} {w.rule} ({w.backend}) L{w.line}: {w.message}")
        out.append("")

    out.append(f"{style.g('cross')} PRE-FLIGHT FAILED - {len(errors) + len(preflight_errors)} "
               f"blocking error(s). Run aborted before compute was allocated.")
    out.append(style.rule())
    return "\n".join(out)


def _render_issue(issue: LintIssue, script: str, style: Style) -> List[str]:
    tpl = get_fix_template(issue.rule)
    title = tpl.title if tpl else issue.rule
    lines = [
        f"{style.g('cross')} STATIC CONFIGURATION ERROR DETECTED",
        f"{style.g('arrow')} Triggered by: {title}",
        f"{style.g('arrow')} Location: {script}, Line {issue.line}",
        "",
        f"[Detail]: {issue.message}",
    ]
    if tpl:
        lines += ["", f"{style.g('warn')} WHY THIS WILL HURT YOUR RUN:", tpl.why]
        lines += ["", f"{style.g('wrench')} ACTIONABLE FIX:", tpl.fix]
    return lines


def _render_preflight_error(msg: str, style: Style) -> List[str]:
    # preflight errors already carry a descriptive message; try to attach a
    # template by scanning for a known error name in the message text.
    tpl = None
    for key in ("DistributedConfigError", "CompiledOptimizerStepConflictError",
                "AmbiguousAssertionError", "UnsupportedTargetError"):
        if key.lower().replace("error", "") in msg.lower() or key in msg:
            tpl = get_fix_template(key)
            break
    lines = [
        f"{style.g('cross')} CONFIGURATION ERROR DETECTED",
        f"[Detail]: {msg}",
    ]
    if tpl:
        lines += ["", f"{style.g('warn')} WHY THIS WILL HURT YOUR RUN:", tpl.why]
        lines += ["", f"{style.g('wrench')} ACTIONABLE FIX:", tpl.fix]
    return lines

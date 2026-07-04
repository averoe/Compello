"""End-to-end Compello demo of the full terminal lifecycle (pure numpy).

Shows the three terminal surfaces exactly as a user would see them:
  1. The Pre-Flight Static Shield (trainlint) catching a fatal config bug.
  2. The live training stream: compact status line that stays quiet while stable
     and expands into boxed Compello Runtime Insight blocks on real events
     (gradient conflict + surgery + momentum grace, proxy vanishing-zone + alpha
     tuning + hysteresis, recovery + weight relaxation).
  3. The Post-Training Capacity & Sensitivity autopsy report.

The per-step violation / gradient-norm / conflict signals here are scripted to
exercise each control loop deterministically; in a real run they come from the
backend adapter. Every number the engine prints is still computed from real
controller state and the real OLS estimator.

Run:  python examples/demo_training.py
"""

from __future__ import annotations

import os

import numpy as np

import compello
from compello import expect, render_preflight_shield
from compello.controller import Controller, ControllerConfig
from compello.insights import InsightEngine
from compello.reports import SensitivityProfiler, render_capacity_report
from compello.report_style import Style
from compello.trainlint import lint_file

STYLE = Style.auto()


def preflight_surface() -> None:
    buggy = os.path.join(os.path.dirname(__file__), "buggy_llm.py")
    issues = lint_file(buggy)
    print(render_preflight_shield(issues, script="fine_tune_llm.py", style=STYLE))


def training_surface() -> None:
    print("\n\n")
    print(STYLE.banner("COMPELLO LIVE TRAINING STREAM  (Vision / proxy-IoU)"))

    iou = expect(compello.wrap(np.zeros(4)), assertion_type="proxy",
                 name="spatial_continuity_iou", alpha=20.0)
    cfg = ControllerConfig(strategy="adaptive_pid", tolerance=0.05, patience=8,
                           weight_ceiling=30.0, max_steps=200, sharpness_g_floor=1e-4)
    controller = Controller(cfg)
    controller.register_assertions([iou])
    controller.states[iou.name].alpha = 20.0  # start sharp, so a vanishing zone can occur

    engine = InsightEngine(
        controller, telemetry="compact", total_steps=cfg.max_steps,
        diagnostics_interval=1, modality="vision", backend="JAX/XLA",
        targets={iou.name: "OutputTarget(segmentation_mask) -> spatial_continuity_iou"},
        style=STYLE,
    )
    profiler = SensitivityProfiler(high_impact_threshold=0.005)

    # scripted-but-realistic signals from the "backend"
    violation = 0.30                 # 1 - IoU shortfall, falling slowly over time
    for step in range(cfg.max_steps):
        violation = max(0.0, violation - 0.0025 + (0.015 if step == 40 else 0.0))
        loss = 0.14 - 0.0004 * step + (0.01 if step in (15, 40) else 0.0)

        # proxy gradient collapses at step 15 -> triggers alpha auto-tune
        proxy_grad = 1e-7 if step == 15 else 5e-3

        # a genuine task/constraint conflict fires at step 40 -> surgery + grace
        grad_conflicts = None
        if step == 40:
            controller.engage_surgery(iou.name, beta1=0.9)  # ~10-step grace window
            grad_conflicts = {iou.name: {"cosine": -0.89, "projected": True,
                                         "layer": "model.layers.28.mlp",
                                         "task_loss": "dice_loss"}}

        result = controller.step(
            {iou.name: violation},
            proxy_grad_norms={iou.name: proxy_grad},
            metric_satisfied={iou.name: violation <= cfg.tolerance},
        )
        profiler.observe(iou.name, weight=controller.states[iou.name].weight,
                         violation=violation, task_metric=max(loss, 0.0))
        si = engine.observe(result, loss=max(loss, 0.0), grad_conflicts=grad_conflicts)

        # print status line at a readable cadence, and always when a block fired
        if si.insights or step % 30 == 0 or result.should_stop:
            print(si.render())
        if result.should_stop:
            print(f"\n>>> training stopped at step {step} (converged={result.converged})")
            break

    print("\n")
    print(render_capacity_report(
        controller, profiler, converged=result.converged,
        compute_summary="0.9 GPU-hours (simulated)", diagnostic_overhead_pct=0.02,
        primitive_labels={iou.name: "Vision Proxy-IoU Primitive"}, style=STYLE,
    ))


def main() -> None:
    preflight_surface()
    training_surface()


if __name__ == "__main__":
    main()

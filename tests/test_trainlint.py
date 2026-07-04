from compello.trainlint import lint_source
from compello.trainlint.core import ERROR


def _rules(issues):
    return {i.rule for i in issues}


def test_detects_pytorch_backend_and_rules():
    src = """
import torch
def train(model, loader, optimizer):
    losses = []
    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
"""
    rules = _rules(lint_source(src))
    assert "zero-grad" in rules
    assert "detached-loss" in rules
    assert "train-eval-mode" in rules


def test_clean_pytorch_loop_has_no_zero_grad_or_detached_issues():
    src = """
import torch
def train(model, loader, optimizer):
    model.train()
    losses = []
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
"""
    rules = _rules(lint_source(src))
    assert "zero-grad" not in rules
    assert "detached-loss" not in rules
    assert "train-eval-mode" not in rules


def test_compiled_optimizer_step_conflict():
    src = """
import torch
def train_step(batch):
    out = model(batch)
    loss.backward()
    optimizer.step()
compiled = torch.compile(train_step)
"""
    issues = lint_source(src)
    assert any(i.rule == "compiled-optimizer-step" and i.severity == ERROR for i in issues)


def test_tensorflow_untracked_variable_and_missing_tape():
    src = """
import tensorflow as tf
@tf.function
def step(self, x):
    self.error_integral += x
    g = tape.gradient(loss, variables)
    return g
"""
    rules = _rules(lint_source(src))
    assert "untracked-variable" in rules
    assert "missing-tape-context" in rules


def test_jax_impure_mutation_and_side_effect():
    src = """
import jax
@jax.jit
def step(state, batch):
    state.count += 1
    print("hi")
    return state
"""
    rules = _rules(lint_source(src))
    assert "impure-mutation" in rules
    assert "untraced-side-effect" in rules


def test_jax_distributed_auto_misconfig():
    src = """
import jax
cfg = {"backend": "jax_native", "distributed": "auto"}
"""
    issues = lint_source(src)
    assert any(i.rule == "distributed-auto-misconfiguration" for i in issues)


def test_ambiguous_assertion_always_on():
    src = """
from compello import expect
expect(raw_tensor, lambda y: y > 0)
"""
    # even with no framework imported, the compello rule fires
    issues = lint_source(src)
    assert any(i.rule == "ambiguous-assertion" for i in issues)


def test_wrapped_assertion_not_flagged():
    src = """
import compello
from compello import expect
expect(compello.wrap(t)[0], "> 0.6")
expect(model.output, "> 0")
"""
    issues = lint_source(src)
    assert not any(i.rule == "ambiguous-assertion" for i in issues)

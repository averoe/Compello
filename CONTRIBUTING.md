# Contributing to Compello

Thank you for your interest in improving Compello. This document explains how to
set up a development environment, the standards contributions are held to, and
the workflow for proposing changes.

## Ways to contribute

- Report bugs and unexpected behavior.
- Improve documentation, examples, or error messages.
- Add or refine assertion types, penalty shapes, or diagnostics.
- Implement or harden backend adapter paths (PyTorch, TensorFlow/Keras, JAX).
- Expand the framework integration test matrix.

## Development setup

```
git clone <your-fork-url>
cd compello
python -m venv .venv
. .venv/Scripts/activate        # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e .[dev]
```

The core and its test suite run without any deep-learning framework installed,
using the numpy reference backend. Install `[torch]`, `[tensorflow]`, or `[jax]`
only when working on those adapters.

## Running the checks

```
pytest -q                         # full test suite
python -m compileall -q compello  # byte-compile all modules (incl. adapters)
trainlint compello                # optional: lint the package itself
```

All tests must pass and the package must byte-compile before a change is
merged. New behavior must come with tests.

## Coding standards

- Target Python 3.9+. Use type hints on public functions and classes.
- Keep the core free of hard dependencies. Only backend adapter modules may
  import a deep-learning framework, and always behind a guarded import that
  raises a clear `BackendNotAvailableError` when the framework is absent.
- No monkeypatching of third-party libraries. Integrate through documented
  hooks, callbacks, or explicit user-called functions.
- Prefer backend-agnostic implementations written against `compello.math`. When
  a mechanism genuinely requires a framework-specific primitive, isolate the
  portable logic into a tested helper and keep the framework call a thin wrapper.
- Match the existing style: descriptive docstrings that reference the relevant
  design section, small focused functions, and explicit error types.
- Do not add emojis to source, documentation, or generated output paths unless a
  feature (such as telemetry glyphs) already uses them intentionally and behind
  an ASCII fallback.

## Testing expectations

- Framework-independent code must be covered by tests that run on the numpy
  backend.
- When you touch a framework adapter, extract any non-trivial logic into a
  backend-agnostic function and test that with numpy. Framework-specific glue
  should be minimal and clearly documented.
- If you add a `trainlint` rule, include both a positive case (the rule fires)
  and a negative case (clean code is not flagged).

## Pull request workflow

1. Create a feature branch from the default branch.
2. Make focused commits with clear messages.
3. Ensure `pytest` passes and the package byte-compiles.
4. Update `CHANGELOG.md` under the Unreleased section.
5. Update documentation (README and docstrings) for any user-facing change.
6. Open a pull request describing the motivation, the change, what you tested,
   and any behavior that could not be verified in your environment.

## Reporting bugs

Open an issue with: the Compello version, Python version, active backend and its
version (if any), a minimal reproduction, the expected behavior, and the actual
behavior including the full error output.

## Security issues

Do not open a public issue for a security vulnerability. Follow the process in
`SECURITY.md`.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, the same license as the project.

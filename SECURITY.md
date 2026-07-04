# Security Policy

## Supported versions

Compello is pre-1.0 software. Security fixes are applied to the latest released
minor version. Until 1.0, older versions are not maintained.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

Please report suspected vulnerabilities privately. Do not open a public issue,
pull request, or discussion for a security problem.

- Use the repository's private vulnerability reporting channel (for example,
  GitHub Security Advisories) where available.
- Otherwise, contact the maintainers through the private contact address listed
  in the project metadata.

In your report, include:

- A description of the issue and its potential impact.
- The Compello version, Python version, and active backend.
- A minimal reproduction or proof of concept.
- Any suggested remediation.

### What to expect

- Acknowledgment of your report as soon as it is triaged.
- An assessment of severity and affected versions.
- A coordinated fix and release, with credit to the reporter unless anonymity is
  requested.

Please allow a reasonable disclosure window before making details public.

## Security considerations when using Compello

Compello runs inside your training process and interacts with model gradients,
optimizer state, and configuration. Keep the following in mind.

- Configuration and code are trusted input. `load_config` parses YAML with a
  safe loader and does not execute arbitrary code, but the `backend` and
  assertion definitions ultimately drive real training code. Only load configs
  from sources you trust.
- `trainlint` is a static analyzer. It parses source with the standard-library
  `ast` module and never imports or executes the target file. It is safe to run
  against untrusted code, but its findings are advisory, not a security
  sandbox.
- Custom assertion types and penalty functions registered through
  `register_assertion_type` execute arbitrary Python during training. Treat them
  as first-party code and review them accordingly.
- Checkpoints carry controller state. The portable JSON and numpy `.npz` formats
  contain only numeric state and are safe to load. The optional `.pt` (torch)
  format uses the framework's own serialization; load `.pt` checkpoints only
  from trusted sources, consistent with the framework's own guidance on
  deserializing untrusted files.
- Distributed synchronization issues collective communication calls on the
  process group you have already initialized. Compello does not open network
  connections of its own.
- Compello does not transmit code, data, or telemetry off the machine. All
  telemetry is written to your own standard output or logs.

## Scope

This policy covers the Compello codebase. Vulnerabilities in third-party
dependencies (PyTorch, TensorFlow, Keras, JAX, Optax, Cooper, NumPy, PyYAML)
should be reported to those projects; if a Compello usage pattern amplifies such
a vulnerability, we still want to hear about it.

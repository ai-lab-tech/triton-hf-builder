# Contributing

Focused bug reports, fixes, documentation and runtime adapters are welcome.
Keep discussions respectful and technical. Explain the problem and resulting
behavior in pull requests, with validation and remaining limitations.

## Development

Use Python 3.10 or newer in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check src scripts tests
ruff format --check src scripts tests
python -m unittest discover -s tests -v
python scripts/check_release.py
python -m build
python -m twine check dist/*
```

CI installs the built wheel on Python 3.10, 3.12 and 3.14 and runs the unit
suite against it. These tests use fake checkpoint weights and mocked engines;
they do not establish real model or GPU compatibility. Real-serving tests are
opt-in: follow [scripts/README.md](scripts/README.md). Report both separately.

## Design and tests

Keep generation independent of inference frameworks, deployment services and
editor APIs. Read metadata without executing model code. Runtime imports belong
in generated templates. Preserve the Python API, JSON CLI contract and relocatable
artifact layout; document compatibility changes to the serving manifest.

Use Arrange–Act–Assert for new and modified tests, with explicit phase comments.
Test observable behavior and failure handling. Never weaken real inference
comparisons to make a model pass. Do not commit weights, credentials, generated
repositories or test reports. Include provenance for third-party code; model
licenses remain separate from this project's Apache-2.0 license.

See [RELEASING.md](RELEASING.md) for packaging and publication.

# Contributing to NeuroFlow

Thanks for your interest in NeuroFlow. This project is at **Stage 1 (MVP)**
moving into **Stage 2 (operator coverage)** — see [`ROADMAP.md`](ROADMAP.md)
for the full picture.

## Code of Conduct

All participants are expected to follow
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Be kind, be technical.

## Ways to contribute

- **Bug reports** — open a [bug issue][bug].
- **Feature requests** — open a [feature issue][feat]. For new operators, IR
  changes, or C++ ABI changes, please use the [RFC template][rfc] first.
- **Pull requests** — bug fixes, tests, docs, new operators, and SDKs all
  welcome. Read the [PR template][pr] before opening.
- **Reviews and triage** — every maintainer started by triaging issues.

[bug]: .github/ISSUE_TEMPLATE/bug_report.md
[feat]: .github/ISSUE_TEMPLATE/feature_request.md
[rfc]: .github/ISSUE_TEMPLATE/rfc.md
[pr]: .github/PULL_REQUEST_TEMPLATE.md

## Development setup

Tested on Windows (MinGW g++ 13.1), macOS (Apple Clang), and Ubuntu 22.04.
Python ≥ 3.10 recommended; the CI matrix covers 3.10 / 3.11 / 3.12.

```bash
git clone https://github.com/<owner>/neuroflow.git
cd neuroflow

# Python
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -U pip
pip install -r requirements.txt
pip install -e ".[dev]"

# C++ runtime (optional, for C++ tests and the pybind11 binding)
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target nflow_core test_runtime

# Run the full test suite
pytest tests/ -v

# Lint
ruff check neuroflow tests examples
```

## Project layout

| Path | Purpose |
|---|---|
| `neuroflow/` | Python research layer (PyTorch models, IR spec, training) |
| `cpp/` | Zero-dep C++ runtime + optional pybind11 binding |
| `tests/` | Pytest suite (currently 8 unit tests, all pure-Python) |
| `examples/` | End-to-end scripts (Burgers 1D training + C++ inference) |
| `paper/` | arXiv-ready LaTeX (Stage 1 MVP paper, 11 pages) |
| `docs/` | Long-form docs, design notes, build matrix |
| `artifacts/` | Pre-trained model artifacts (release-only, not in git) |

## Style

- **Python:** `ruff` config in `pyproject.toml`. `mypy --strict` is the goal,
  not yet enforced everywhere. Public functions take a `Config` dataclass,
  not a giant `**kwargs`.
- **C++:** C++17, zero third-party dependencies in the runtime core. Pybind11
  is an opt-in binding, gated behind a CMake option.
- **IR:** NeuroIR v0 is a frozen wire format. Any change that breaks v0
  triggers a v1 bump and goes through the [RFC process][rfc].

## Commit and PR conventions

- Imperative mood in commit subjects: *"Add SpectralConv2d"*, not *"Added"*.
- One logical change per commit. Squash fixups locally; we don't squash-merge.
- Reference the issue number in the commit subject when applicable
  (`#123`).
- PR descriptions must use the template. *"Drive-by fix"* is fine in the
  title; in the body, say what changed and how you tested.

## Review process

1. Open a PR; CI must be green before review.
2. A maintainer reviews within ~3 working days. We may request changes;
   we may also merge immediately if the change is small and well-tested.
3. Squash-merge is off by default. We use the GitHub "Rebase and merge"
   strategy to keep history linear.

## Release process

Releases are tagged with `vX.Y.Z` (semver). Each tag:

- Triggers a CI build across the OS × Python matrix.
- Publishes a draft GitHub Release with pre-built C++ binaries
  (Linux/macOS/Windows) attached.
- Triggers a Zenodo snapshot and produces a new DOI.
- Updates the `CHANGELOG.md` "Unreleased" → versioned section.

## License

By contributing, you agree that your contributions will be licensed under
the [Apache License 2.0](LICENSE).

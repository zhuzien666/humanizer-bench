# Contributing to humanizer-bench

Thanks for your interest! This project is built to be extended.

## Development setup

```bash
git clone https://github.com/zhuzien666/humanizer-bench.git
cd humanizer-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Workflow

We use a standard fork-and-pull-request workflow:

1. Fork the repository (or create a feature branch if you have write access).
2. Make your change with tests.
3. Open a pull request describing what and why.
4. A maintainer reviews and merges.

This keeps `main` reviewable and gives every change a clear, attributable history —
which matters for a research tool.

## Adding a component

Each of the four parts has a base class with a single required method:

| Part | Base class | Implement | Add tests in |
|------|-----------|-----------|--------------|
| Detector | `BaseDetector` | `score(text) -> float` in `[0, 1]` | `tests/test_detectors.py` |
| Attack   | `BaseAttack`   | `transform(text) -> str`            | `tests/test_attacks.py`   |
| Dataset  | `BaseDataset`  | `__iter__() -> Iterator[Example]`   | `tests/test_datasets.py`  |

Please:

- Add a docstring to every public class and function.
- Add at least one test per new component.
- Keep `pytest` green (`pytest` from the repo root).

## Style

- Follow PEP 8; keep functions small and documented.
- Prefer the standard library for the core; heavyweight model dependencies belong behind
  the `models` optional-dependency extra.

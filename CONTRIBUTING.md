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
- Export the component from its package `__init__.py` **and** register it in
  `humanizer_bench/registry.py` so the CLI and `humanizer-bench --list` can find it.
- Add at least one test per new component.
- Keep `pytest` green (`pytest` from the repo root).

## Model-backed components

Detectors that load a language model (and attacks that translate text) need `torch` and
`transformers`. Those live behind the `models` extra so the core install stays light:

```bash
pip install -e ".[models]"
```

Two conventions keep such components friendly to users who have *not* installed the extra:

**1. Import through `require`, inside the method that needs it.** Lazy-importing keeps
construction cheap and turns a missing install into one actionable message:

```python
from .._deps import require


class MyModelDetector(BaseDetector):
    name = "my-model"

    def score(self, text: str) -> float:
        transformers = require("transformers")   # ImportError names the install command
        ...
```

Cache anything expensive (a loaded model or tokenizer) on the instance so it is built
once, not once per call.

**2. Mark tests that need a real model.** CI installs only `[dev]`, so a test that loads a
model must opt out cleanly rather than fail. Mark it, and `tests/conftest.py` skips it
whenever the extra is absent:

```python
import pytest


@pytest.mark.models
def test_scores_ai_text_above_human_text():
    ...
```

The marker also lets you split the suite while developing — `pytest -m "not models"` runs
the fast tests without loading anything, `pytest -m models` runs only the slow ones.

Keep at least one test that runs *without* the extra — for example, asserting that the
component is registered, or that constructing it does not import the heavy dependency.

**3. Declare new heavy dependencies.** If your component needs a library that the `models`
extra does not already list, add it there. A missing transitive dependency (a tokenizer
backend, say) surfaces as a confusing failure deep inside `transformers`, not as a clear
"please install X".

## Style

- Follow PEP 8; keep functions small and documented.
- Prefer the standard library for the core; heavyweight model dependencies belong behind
  the `models` optional-dependency extra.

---
name: New component (detector / attack / dataset)
about: Propose or claim a new detector, attack, or dataset
title: ''
labels: enhancement
---

## Component

- **Type**: detector / attack / dataset
- **Name** (the `name` attribute used on the CLI):
- **Upstream method or corpus** (paper, model, dataset link):

## What it does

<!-- One paragraph: what signal does it use, or how does it rewrite text? -->

## Why add it

<!-- What does it let the benchmark measure that we cannot measure today? -->

## Checklist

- [ ] Subclasses the relevant base class and implements its one required method
- [ ] Exported from the package `__init__.py` and registered in `humanizer_bench/registry.py`
- [ ] Tests added; model-backed tests are marked `@pytest.mark.models` (tests/conftest.py skips them when the extra is absent)
- [ ] Any new heavy dependency added to the `models` extra in pyproject.toml
- [ ] Heavy dependencies imported lazily via `_deps.require()`
- [ ] README component table updated

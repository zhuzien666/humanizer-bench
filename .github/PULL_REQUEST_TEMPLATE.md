<!--
Keep this short but complete. A reviewer should understand the change without
reading the diff first. Delete any section that genuinely does not apply.
-->

## What

<!-- What does this PR change? One or two sentences. -->

## Why

<!-- Why is this change needed? Link the issue: "Closes #12" -->

Closes #

## Testing

<!--
How did you verify this works? For example:
- `pytest -q` passes locally (N tests)
- manual run: `humanizer-bench --list` shows the new component
-->

## Checklist

- [ ] `pytest -q` passes locally
- [ ] New public functions/classes have docstrings
- [ ] New component (if any) is exported in its package `__init__.py` **and** registered in `humanizer_bench/registry.py`
- [ ] Docs updated if user-facing behaviour changed (README / CONTRIBUTING / examples)

"""Shared pytest configuration.

Skips tests marked ``@pytest.mark.models`` when the optional ``models`` extra
(torch/transformers) is not installed, so the core suite stays runnable
without the heavy deps -- e.g. in CI's default ``pip install -e ".[dev]"`` job.
"""

import pytest

try:
    import torch  # noqa: F401

    _HAS_MODELS = True
except ImportError:
    _HAS_MODELS = False


def pytest_collection_modifyitems(config, items):
    if _HAS_MODELS:
        return
    skip_models = pytest.mark.skip(reason="requires the 'models' extra (torch/transformers)")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip_models)

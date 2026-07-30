"""Helpers for optional, heavyweight dependencies.

The core package runs on the standard library alone. Model-backed components --
detectors that load a language model, attacks that translate text -- need
``torch``/``transformers``, which live behind the ``models`` extra so that
installing humanizer-bench stays cheap for users who only want the framework.

Import those libraries through :func:`require` so that a missing install produces
one clear, actionable message instead of a bare ``ImportError`` from deep inside
a component.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str = "models") -> ModuleType:
    """Import and return ``module``, or explain how to install it.

    Args:
        module: Import path of the optional dependency, e.g. ``"transformers"``.
        extra: Name of the optional-dependency group that provides it.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed. The message names the
            missing package and the exact command that installs it.

    Example:
        Import lazily inside the method that needs it, so constructing the
        component (and importing the package) stays cheap::

            from .._deps import require

            class MyModelDetector(BaseDetector):
                name = "my-model"

                def score(self, text: str) -> float:
                    transformers = require("transformers")
                    ...
    """
    try:
        return importlib.import_module(module)
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{module!r} is needed for this component but is not installed. "
            f"Install the optional dependencies with:\n\n"
            f'    pip install -e ".[{extra}]"\n'
        ) from err

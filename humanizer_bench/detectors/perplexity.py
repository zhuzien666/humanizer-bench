"""GPT-2 perplexity detector (Phase 2 -- not yet implemented).

The real implementation will load a small causal language model (e.g.
``distilgpt2``) and compute token-level perplexity, treating low perplexity as
evidence of machine generation. It is kept as a stub so the package layout and
import paths are stable while the student implements it.
"""

from __future__ import annotations

from .base import BaseDetector


class PerplexityDetector(BaseDetector):
    """Placeholder for a GPT-2 perplexity-based detector."""

    name = "perplexity"

    def __init__(self, model_name: str = "distilgpt2") -> None:
        self.model_name = model_name

    def score(self, text: str) -> float:  # pragma: no cover - stub
        raise NotImplementedError(
            "PerplexityDetector is a Phase 2 stub. Load a causal LM with "
            "`transformers`, compute token-level perplexity, and map it to "
            "[0, 1]. Install extras with `pip install -e '.[models]'`."
        )

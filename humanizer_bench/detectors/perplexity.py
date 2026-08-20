"""GPT-2 perplexity detector.

Loads a small causal language model (``distilgpt2`` by default) and computes
token-level perplexity: how "surprised" the model is by the text, on average,
per token. Machine-generated text tends to be more predictable to a language
model (low perplexity), while human writing tends to be less predictable
(high perplexity). We map perplexity through a logistic so low perplexity
maps to a high P(AI), producing a probability in ``[0, 1]``.

This detector requires the ``models`` extra (``pip install -e '.[models]'``)
for ``torch`` and ``transformers``. Those imports are deferred to
:meth:`PerplexityDetector._load` so importing this module -- or the rest of
the package -- never requires the heavy deps to be installed.
"""

from __future__ import annotations

import math
import re

from .._deps import require
from .base import BaseDetector

_WORD = re.compile(r"[A-Za-z\']+")

#: Rough "typical prose" perplexity under distilgpt2; below this looks
#: AI-like, above looks human. A crude, dataset-independent prior, not a
#: fitted parameter -- same spirit as the heuristic detector's reference.
_REF_PERPLEXITY = 40.0
#: Logistic slope -- how sharply the score reacts to the perplexity gap.
_SLOPE = 0.08


class PerplexityDetector(BaseDetector):
    """Score text from GPT-2 token-level perplexity."""

    name = "perplexity"

    def __init__(self, model_name: str = "distilbert/distilgpt2") -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        """Import torch/transformers and load the model, once, on first use."""
        if self._model is not None:
            return

        torch = require("torch")
        transformers = require("transformers")

        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()

    def score(self, text: str) -> float:
        # Mirror HeuristicDetector: too little signal to say anything, so
        # stay neutral rather than guess.
        if len(_WORD.findall(text)) < 5:
            return 0.5

        self._load()
        torch = self._torch

        encodings = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings["input_ids"]

        if input_ids.shape[1] < 2:
            return 0.5

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            mean_neg_log_likelihood = outputs.loss.item()

        perplexity = math.exp(mean_neg_log_likelihood)

        # Low perplexity -> positive logit -> higher P(AI).
        logit = _SLOPE * (_REF_PERPLEXITY - perplexity)
        return 1.0 / (1.0 + math.exp(-logit))

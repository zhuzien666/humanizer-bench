"""A zero-dependency lexical baseline detector.

This detector is intentionally simple: it exists so the end-to-end pipeline is
runnable with **no model downloads**. It is *not* a serious detector. Replace it
with :class:`~humanizer_bench.detectors.perplexity.PerplexityDetector` (GPT-2) or
a Binoculars wrapper in Phase 2.

The signal it uses is **burstiness** -- the variation in sentence length. Machine
text tends to have uniform, evenly-paced sentences (low burstiness), whereas human
writing tends to mix long and short sentences (high burstiness). We squash the
burstiness through a logistic so the output is a probability in ``[0, 1]``. The
reference point and slope are crude, dataset-independent priors, not fitted
parameters -- so the detector is deliberately mediocre.
"""

from __future__ import annotations

import math
import re
from statistics import pstdev

from .base import BaseDetector

_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[.!?]+")

#: Rough "typical prose" burstiness; below this looks AI-like, above looks human.
_REF_BURSTINESS = 0.35
#: Logistic slope -- how sharply the score reacts to the burstiness gap.
_SLOPE = 6.0


class HeuristicDetector(BaseDetector):
    """Score text from sentence-length burstiness (a placeholder baseline)."""

    name = "heuristic"

    def score(self, text: str) -> float:
        sentence_lengths = [
            len(_WORD.findall(s)) for s in _SENT.split(text) if s.strip()
        ]
        # Need at least two sentences and enough words to estimate burstiness.
        if len(sentence_lengths) < 2 or sum(sentence_lengths) < 5:
            return 0.5

        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        burstiness = pstdev(sentence_lengths) / mean_length if mean_length else 0.0

        # Low burstiness -> positive logit -> higher P(AI).
        logit = _SLOPE * (_REF_BURSTINESS - burstiness)
        return 1.0 / (1.0 + math.exp(-logit))

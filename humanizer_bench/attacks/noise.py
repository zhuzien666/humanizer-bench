"""A simple, dependency-free perturbation attack.

Applies light surface noise -- character drops, duplications, and case swaps --
to mimic the crude edits a naive "humanizer" might make. This runs with no model
downloads so the pipeline works out of the box.

Note: this attack changes *characters*, not sentence structure, so it targets
character- and perplexity-level detectors. It will barely move a burstiness-based
detector like :class:`~humanizer_bench.detectors.heuristic.HeuristicDetector` --
a useful reminder that an attack's effectiveness depends on which detector it
faces. Pair it with the GPT-2 perplexity detector once that lands in Phase 2.
"""

from __future__ import annotations

import random

from .base import BaseAttack


class NoiseAttack(BaseAttack):
    """Randomly perturb a fraction of alphabetic characters.

    Args:
        rate: Probability that any given letter is perturbed.
        seed: Optional RNG seed for reproducibility.
    """

    name = "noise"

    def __init__(self, rate: float = 0.05, seed: int | None = 0) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        self.rate = rate
        self._rng = random.Random(seed)

    def transform(self, text: str) -> str:
        out = []
        for ch in text:
            if ch.isalpha() and self._rng.random() < self.rate:
                choice = self._rng.randint(0, 2)
                if choice == 0:
                    continue  # drop the character
                elif choice == 1:
                    out.append(ch * 2)  # duplicate it
                else:
                    out.append(ch.swapcase())  # flip case
            else:
                out.append(ch)
        return "".join(out)

"""Minimal end-to-end benchmark loop (Phase 1).

Runs one detector over one dataset, before and after one attack, and prints an
accuracy comparison. This is the "closed loop" that the rest of the project
generalizes: swap in more detectors, attacks, datasets, and metrics, and the
loop stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attacks.base import BaseAttack
from .attacks.sentence_merge import SentenceMergeAttack
from .datasets.base import BaseDataset
from .datasets.toy import ToyDataset
from .detectors.base import BaseDetector
from .detectors.heuristic import HeuristicDetector
from .metrics.core import accuracy, false_positive_rate


@dataclass
class RunResult:
    """Outcome of a single detector/attack/dataset evaluation."""

    detector: str
    attack: str
    n: int
    acc_clean: float
    acc_attacked: float
    fpr_clean: float

    @property
    def drop(self) -> float:
        """Accuracy lost because of the attack (higher = more vulnerable)."""
        return self.acc_clean - self.acc_attacked


def run(
    dataset: BaseDataset | None = None,
    attack: BaseAttack | None = None,
    detector: BaseDetector | None = None,
    threshold: float = 0.5,
) -> RunResult:
    """Evaluate ``detector`` on ``dataset`` before and after ``attack``.

    The attack is applied only to AI-labeled texts (the realistic threat model:
    an adversary launders machine text); human texts are left untouched so the
    false-positive rate stays meaningful. Defaults wire up the fully offline
    toy pipeline.
    """
    dataset = dataset if dataset is not None else ToyDataset()
    attack = attack if attack is not None else SentenceMergeAttack()
    detector = detector if detector is not None else HeuristicDetector()

    examples = list(dataset)
    y_true = [ex.label for ex in examples]

    clean_pred = [detector.predict(ex.text, threshold) for ex in examples]
    attacked_pred = [
        detector.predict(attack(ex.text) if ex.label == 1 else ex.text, threshold)
        for ex in examples
    ]

    return RunResult(
        detector=detector.name,
        attack=attack.name,
        n=len(examples),
        acc_clean=accuracy(y_true, clean_pred),
        acc_attacked=accuracy(y_true, attacked_pred),
        fpr_clean=false_positive_rate(y_true, clean_pred),
    )


def format_result(result: RunResult) -> str:
    """Render a :class:`RunResult` as a small fixed-width table."""
    rows = [
        f"detector={result.detector}   attack={result.attack}   n={result.n}",
        "-" * 44,
        f"{'metric':<26}{'value':>10}",
        f"{'accuracy (clean)':<26}{result.acc_clean:>10.3f}",
        f"{'accuracy (attacked)':<26}{result.acc_attacked:>10.3f}",
        f"{'accuracy drop':<26}{result.drop:>10.3f}",
        f"{'FPR on human (clean)':<26}{result.fpr_clean:>10.3f}",
    ]
    return "\n".join(rows)


def main() -> None:
    """Console-script entry point: run the default toy benchmark and print it."""
    print(format_result(run()))


if __name__ == "__main__":
    main()

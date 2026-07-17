"""Benchmark loops: single runs and detector × attack matrices.

:func:`run` evaluates one detector on one dataset before and after one attack.
:func:`run_matrix` generalizes it to a grid — the shape the eventual heatmap
figures are built from. Swap in more detectors, attacks, datasets, and metrics;
the loops stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

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


def run_matrix(
    detectors: Sequence[BaseDetector],
    attacks: Sequence[BaseAttack],
    dataset: Optional[BaseDataset] = None,
    threshold: float = 0.5,
) -> List[RunResult]:
    """Evaluate every detector × attack pair on one dataset.

    Attacks are applied only to AI-labeled texts (the realistic threat model:
    an adversary launders machine text); human texts pass through untouched so
    the false-positive rate stays meaningful.

    Each attack rewrites the corpus exactly once and the rewritten texts are
    shared across detectors — so every detector is judged on identical inputs,
    and expensive attacks (e.g. back-translation) don't rerun per detector.
    Defaults to the fully offline :class:`ToyDataset`.
    """
    dataset = dataset if dataset is not None else ToyDataset()
    examples = list(dataset)
    y_true = [ex.label for ex in examples]

    # Indexed by position, not attack.name, so duplicate names can't collide.
    attacked_texts: List[List[str]] = [
        [attack(ex.text) if ex.label == 1 else ex.text for ex in examples]
        for attack in attacks
    ]

    results: List[RunResult] = []
    for detector in detectors:
        clean_pred = [detector.predict(ex.text, threshold) for ex in examples]
        acc_clean = accuracy(y_true, clean_pred)
        fpr_clean = false_positive_rate(y_true, clean_pred)
        for attack, texts in zip(attacks, attacked_texts):
            attacked_pred = [detector.predict(text, threshold) for text in texts]
            results.append(
                RunResult(
                    detector=detector.name,
                    attack=attack.name,
                    n=len(examples),
                    acc_clean=acc_clean,
                    acc_attacked=accuracy(y_true, attacked_pred),
                    fpr_clean=fpr_clean,
                )
            )
    return results


def run(
    dataset: Optional[BaseDataset] = None,
    attack: Optional[BaseAttack] = None,
    detector: Optional[BaseDetector] = None,
    threshold: float = 0.5,
) -> RunResult:
    """Evaluate one detector on one dataset before and after one attack.

    A thin wrapper over :func:`run_matrix` for the 1×1 case. Defaults wire up
    the fully offline toy pipeline.
    """
    attack = attack if attack is not None else SentenceMergeAttack()
    detector = detector if detector is not None else HeuristicDetector()
    return run_matrix([detector], [attack], dataset=dataset, threshold=threshold)[0]


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


def format_matrix(results: Sequence[RunResult]) -> str:
    """Render matrix results as a grid: one detector per row, one attack per column.

    Cells show the attack-induced accuracy drop (higher = more vulnerable);
    ``clean`` and ``FPR`` columns give each detector's no-attack baseline.
    """
    if not results:
        return "(no results)"

    detectors: List[str] = []
    attacks: List[str] = []
    cell = {}
    for r in results:
        if r.detector not in detectors:
            detectors.append(r.detector)
        if r.attack not in attacks:
            attacks.append(r.attack)
        cell[(r.detector, r.attack)] = r

    det_w = max(len("detector"), max(len(d) for d in detectors)) + 2
    col_ws = [max(len(a), 6) + 2 for a in attacks]

    lines = [
        f"n={results[0].n}   cells: accuracy drop after attack (higher = more vulnerable)",
        f"{'detector':<{det_w}}{'clean':>7}{'FPR':>7}  "
        + "".join(f"{a:>{w}}" for a, w in zip(attacks, col_ws)),
    ]
    for d in detectors:
        base = next(cell[(d, a)] for a in attacks if (d, a) in cell)
        row = f"{d:<{det_w}}{base.acc_clean:>7.3f}{base.fpr_clean:>7.3f}  "
        for a, w in zip(attacks, col_ws):
            r = cell.get((d, a))
            row += f"{r.drop:>{w}.3f}" if r is not None else f"{'--':>{w}}"
        lines.append(row)
    return "\n".join(lines)

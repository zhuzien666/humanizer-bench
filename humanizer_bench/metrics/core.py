"""Evaluation metrics for detector benchmarking.

Phase 1 ships the essentials needed by the minimal loop. Phase 2 adds ROC-AUC,
precision/recall/F1, and bootstrap confidence intervals on every metric.
"""

from __future__ import annotations

from typing import Sequence


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of predictions that match the ground-truth labels."""
    if not y_true:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return correct / len(y_true)


def false_positive_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of human texts (label ``0``) wrongly flagged as AI (pred ``1``).

    This is the headline fairness metric: a high FPR means real human writers --
    including non-native English speakers -- get falsely accused.
    """
    negatives = [(a, b) for a, b in zip(y_true, y_pred) if a == 0]
    if not negatives:
        return 0.0
    return sum(1 for _, b in negatives if b == 1) / len(negatives)

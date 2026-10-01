"""Evaluation metrics for detector benchmarking.

Phase 1 ships the essentials needed by the minimal loop. Phase 2 adds ROC-AUC,
precision/recall/F1, and bootstrap confidence intervals on every metric.

Convention: label ``1`` is the positive class (AI-written text), label ``0``
is the negative class (human-written text). All label-based metrics treat
``y_pred`` as hard labels; :func:`roc_auc` takes raw scores instead.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, Tuple


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


def _confusion_counts(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` for the positive class (label ``1``)."""
    tp = fp = fn = 0
    for a, b in zip(y_true, y_pred):
        if b == 1 and a == 1:
            tp += 1
        elif b == 1 and a == 0:
            fp += 1
        elif b == 0 and a == 1:
            fn += 1
    return tp, fp, fn


def precision(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of flagged texts (pred ``1``) that are actually AI (label ``1``).

    Returns ``0.0`` when nothing is flagged, so a silent detector scores zero.
    """
    tp, fp, _ = _confusion_counts(y_true, y_pred)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of AI texts (label ``1``) the detector catches.

    Returns ``0.0`` when there are no positive examples.
    """
    tp, _, fn = _confusion_counts(y_true, y_pred)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Harmonic mean of :func:`precision` and :func:`recall`.

    Returns ``0.0`` when both are zero (nothing correct to be precise about).
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def roc_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """Area under the ROC curve, via the Mann-Whitney U rank-sum.

    Equivalent to ``P(score(positive) > score(negative))`` with ties counted
    as one half. Takes raw scores (not hard labels) so detectors that output
    probabilities can be ranked. Returns ``0.5`` when there is nothing to
    rank (no positives or no negatives, or an empty input).

    Raises:
        ValueError: If ``y_true`` contains labels other than 0 and 1.
    """
    labels = {int(t) for t in y_true}
    if not labels.issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0/1 labels, got {sorted(labels)}")
    pairs = [(float(s), int(t)) for s, t in zip(y_scores, y_true)]
    n_pos = sum(1 for _, t in pairs if t == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ordered = sorted(pairs, key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    n = len(ordered)
    while i < n:
        # 1-based average rank over the tie group [i, j).
        j = i
        while j < n and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if ordered[k][1] == 1:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_ci(
    metric: Callable[[Sequence[int], Sequence[float]], float],
    y_true: Sequence[int],
    y_pred_or_scores: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for any paired metric.

    ``metric`` is any ``f(y_true, y_pred_or_scores) -> float`` (e.g.
    :func:`accuracy`, :func:`precision`, :func:`false_positive_rate`;
    :func:`roc_auc` works too with scores passed in place of hard labels).
    Resamples ``n`` examples with replacement ``n_bootstrap`` times and
    returns the ``(lower, upper)`` percentiles of the resampled metric
    values. Deterministic for a fixed ``seed``.
    """
    n = len(y_true)
    if len(y_true) != len(y_pred_or_scores):
        raise ValueError("y_true and y_pred_or_scores must have the same length.")
    if n == 0:
        return (0.0, 0.0)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be strictly between 0.0 and 1.0.")
    rng = random.Random(seed)
    stats = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(
            metric([y_true[i] for i in idx], [y_pred_or_scores[i] for i in idx])
        )
    stats.sort()
    alpha = 1.0 - ci
    lo_pos = max(0, min(int(alpha / 2 * n_bootstrap), n_bootstrap - 1))
    hi_pos = max(0, min(int((1 - alpha / 2) * n_bootstrap), n_bootstrap - 1))
    return (stats[lo_pos], stats[hi_pos])

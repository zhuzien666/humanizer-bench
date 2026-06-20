"""humanizer-bench: a benchmark for the robustness of AI-text detectors.

The package is organized around four pluggable abstractions:

* :class:`~humanizer_bench.detectors.base.BaseDetector` -- scores text as AI vs. human
* :class:`~humanizer_bench.attacks.base.BaseAttack`     -- rewrites text to evade detection
* :class:`~humanizer_bench.datasets.base.BaseDataset`   -- yields labeled examples
* metrics (:mod:`humanizer_bench.metrics.core`)         -- turn predictions into scores

See :func:`humanizer_bench.runner.run` for the minimal end-to-end loop.
"""

__version__ = "0.0.1"

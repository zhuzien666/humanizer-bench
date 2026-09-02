# Architecture

humanizer-bench is built around four small interfaces. Everything else — the CLI, the
evaluation loops, the result tables — is glue that only ever talks to these four
contracts. That is what makes a new detector (or attack, or dataset) a ~40-line subclass
rather than a change to the framework.

## The pipeline

```
  ① Dataset  ──▶  Example(text, label, source)
                      │
                      ▼
  ② Attack   ──▶  rewritten text            (applied to AI-labeled texts only)
                      │
                      ▼
  ③ Detector ──▶  P(text is AI) ∈ [0, 1]
                      │
                      ▼
  ④ Metrics  ──▶  accuracy, FPR, attack-induced drop
```

## The four contracts

| Part | Base class | Implement | Contract |
|------|-----------|-----------|----------|
| Dataset | `BaseDataset` | `__iter__()` | Yields `Example`s. Must be **repeatable**: iterating twice yields the same data. |
| Attack | `BaseAttack` | `transform(text) -> str` | Returns a rewritten string. Should preserve meaning; may be stochastic, but must accept a `seed` for reproducibility. |
| Detector | `BaseDetector` | `score(text) -> float` | Returns `P(AI)` in `[0, 1]`. **Higher = more likely machine-generated.** `predict()` and `score_batch()` come from the base class. |
| Metrics | — (plain functions) | — | `metrics/core.py`; take `(y_true, y_pred)` sequences. |

Two conventions that are easy to miss:

- **`0.5` means "I don't know."** Detectors return `0.5` when there is too little signal
  (e.g. a one-token input). Returning `0` or `1` would be claiming certainty from no
  evidence — the wrong failure mode for a tool that can falsely accuse a human writer.
- **Attacks only touch AI-labeled texts.** This is the realistic threat model: an
  adversary launders machine text. Human texts pass through untouched, which keeps the
  false-positive rate meaningful before and after an attack.

## Evaluation loops

`runner.py` has two entry points:

- **`run(dataset, attack, detector)`** — one detector, one attack. Returns a `RunResult`
  with clean accuracy, attacked accuracy, their difference (`drop`), and clean FPR.
- **`run_matrix(detectors, attacks, dataset)`** — every detector × attack pair.

`run()` is a thin 1×1 wrapper over `run_matrix()`, so there is exactly one evaluation
code path. Inside `run_matrix`, each attack rewrites the corpus **once** and the
rewritten texts are shared across detectors. Two reasons: every detector is judged on
byte-identical inputs, and an expensive attack (back-translation, paraphrasing) does not
re-run per detector.

## Registry and CLI

`registry.py` maps short names to classes (`"heuristic"` → `HeuristicDetector`). The CLI
resolves `--detector`/`--attack`/`--dataset` names through it, so a registered component
is immediately available from the command line and in `humanizer-bench --list`. A new
component is not "done" until it is both exported from its package `__init__.py` and
listed in the registry.

`--json` emits the same results as a machine-readable object instead of a text table, so
downstream analysis (the eventual heatmap figure) does not have to parse the aligned
columns. Each entry carries `drop` explicitly rather than leaving it to be recomputed —
`dataclasses.asdict` would omit it, since it is a property on `RunResult`.

## Optional dependencies

The core package imports nothing outside the standard library, so `pip install
humanizer-bench` stays cheap and the offline demo always runs. Components that load a
language model import `torch`/`transformers` lazily through
`humanizer_bench._deps.require()`, which turns a missing install into one actionable
message. See [CONTRIBUTING.md](../CONTRIBUTING.md#model-backed-components).

## Why this shape

The benchmark's central claim is that **robustness is attack × detector specific** — an
attack degrades only the detectors that depend on the signal it destroys. Character-level
noise wrecks a perplexity detector while leaving a sentence-rhythm detector untouched;
sentence restructuring does the reverse. Measuring that requires running every detector
against every attack on identical data, which is exactly what the matrix loop and the
four interchangeable contracts exist to make cheap.

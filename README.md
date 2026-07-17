# humanizer-bench

> A benchmark for the **robustness of AI-text detectors** against "humanizer" attacks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status: early development](https://img.shields.io/badge/status-early%20development-orange)

**humanizer-bench** is a small, extensible Python toolkit for measuring how well AI-text
detectors (GPTZero-style tools, Binoculars, Fast-DetectGPT, …) hold up when text is
deliberately rewritten to evade them — and how often they falsely accuse *real human
writers*, including non-native English speakers.

> ⚠️ **Early development.** The architecture and a fully-offline minimal pipeline are in
> place. Real detectors, attacks, and dataset loaders are being added (see
> [Roadmap](#roadmap)).

## The idea: four pluggable parts

Everything is built around four interfaces, so adding a new detector (or attack, or
dataset) is just writing one small subclass:

```
  ① Dataset  ──▶ labeled text (AI / human)
       │
       ▼
  ② Attack   ──▶ rewritten text (tries to evade detection)
       │
       ▼
  ③ Detector ──▶ P(text is AI) in [0, 1]
       │
       ▼
  ④ Metrics  ──▶ accuracy, FPR, attack-induced drop, …
```

| Part | Base class | Ships now | Coming (Phase 2) |
|------|-----------|-----------|------------------|
| Detector | `BaseDetector` | `HeuristicDetector` (burstiness baseline) | GPT-2 perplexity, Binoculars, Fast-DetectGPT |
| Attack   | `BaseAttack`   | `SentenceMergeAttack`, `NoiseAttack`   | back-translation, synonym substitution |
| Dataset  | `BaseDataset`  | `ToyDataset` (bundled, offline)        | RAID, human control, non-native English |
| Metrics  | —              | accuracy, false-positive rate          | ROC-AUC, P/R/F1, bootstrap CIs |

> The default pairing (`HeuristicDetector` × `SentenceMergeAttack`) is chosen so the
> demo shows a real attack-induced accuracy drop. `NoiseAttack` perturbs characters
> instead of structure, so it barely affects the burstiness detector — a built-in
> reminder that robustness is *attack × detector specific*, which is the whole point
> of the benchmark.

## Quickstart

No dependencies are required to run the offline demo — it uses only the standard library.

```bash
git clone https://github.com/zhuzien666/humanizer-bench.git
cd humanizer-bench
python run.py
```

You'll get a before/after table like:

```
detector=heuristic   attack=sentence_merge   n=18
--------------------------------------------
metric                         value
accuracy (clean)               0.833
accuracy (attacked)            0.500
accuracy drop                  0.333
FPR on human (clean)           0.333
```

Once installed (`pip install -e .`), the same loop is available as the
`humanizer-bench` command. Repeat `--detector`/`--attack` to get a full
detector × attack grid:

```bash
humanizer-bench --list                        # registered components
humanizer-bench -a sentence_merge -a noise    # matrix: one column per attack
```

```
n=18   cells: accuracy drop after attack (higher = more vulnerable)
detector     clean    FPR    sentence_merge   noise
heuristic    0.833  0.333             0.333   0.000
```

Or from Python:

```python
from humanizer_bench.runner import run, format_result
from humanizer_bench.detectors import HeuristicDetector
from humanizer_bench.attacks import SentenceMergeAttack
from humanizer_bench.datasets import ToyDataset

result = run(ToyDataset(), SentenceMergeAttack(rate=1.0), HeuristicDetector())
print(format_result(result))
```

## Install (for development)

```bash
pip install -e ".[dev]"   # core + test tooling
pytest                     # run the test suite
```

Phase 2 model-backed components will live behind an extra:

```bash
pip install -e ".[models]"  # torch, transformers, datasets, …
```

## Extending it

Add a detector by subclassing `BaseDetector` and implementing one method:

```python
from humanizer_bench.detectors import BaseDetector

class MyDetector(BaseDetector):
    name = "my-detector"

    def score(self, text: str) -> float:
        # return P(text is AI-generated) in [0, 1]
        ...
```

Attacks (`BaseAttack.transform`) and datasets (`BaseDataset.__iter__`) follow the same
pattern. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Roadmap

- [x] Four-part architecture + base classes
- [x] Offline minimal closed loop (`python run.py`)
- [x] Initial test suite
- [ ] GPT-2 perplexity detector; Binoculars / Fast-DetectGPT wrappers
- [ ] Back-translation and synonym-substitution attacks
- [ ] Real dataset loaders (RAID + non-native-English corpus)
- [ ] ROC-AUC, P/R/F1, bootstrap confidence intervals
- [x] CLI with detector × attack matrix evaluation
- [x] GitHub Actions CI
- [ ] Docs site

## License

[MIT](LICENSE).

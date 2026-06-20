# Quickstart

The fastest way to see humanizer-bench work end to end. No dependencies needed —
the toy pipeline uses only the Python standard library.

## Run the default benchmark

```bash
python run.py
```

## Compose your own run

```python
from humanizer_bench.runner import run, format_result
from humanizer_bench.detectors import HeuristicDetector
from humanizer_bench.attacks import SentenceMergeAttack
from humanizer_bench.datasets import ToyDataset

result = run(
    dataset=ToyDataset(),
    attack=SentenceMergeAttack(rate=1.0, seed=0),
    detector=HeuristicDetector(),
    threshold=0.5,
)

print(format_result(result))
print("attack-induced accuracy drop:", result.drop)
```

## What to try next

- Swap in `NoiseAttack` and notice it barely dents the burstiness detector — an
  attack only helps against the detector whose weakness it targets.
- Write your own `BaseDetector` subclass and pass it to `run(...)`.
- (Phase 2) Swap `HeuristicDetector` for the GPT-2 `PerplexityDetector`, and add
  `BackTranslationAttack`, once those are implemented.

> A runnable Jupyter notebook version of this walkthrough will live here as
> `quickstart.ipynb` once the model-backed components land.

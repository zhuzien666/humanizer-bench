# False-positive bias on non-native English writing

The project's differentiator: do AI-text detectors falsely flag non-native
English writers as AI? This walkthrough measures the **false-positive rate
(FPR)** of each detector on the JFLEG corpus — sentences written by
English-language learners, all human-labeled.

## The corpus

[`jhu-clsp/jfleg`](https://huggingface.co/datasets/jhu-clsp/jfleg) on the
HuggingFace Hub: 1,511 sentences from the JFLEG grammatical-error-correction
benchmark (Napoles et al., 2017), written by learners with different first
languages. We use the original `sentence` field — the learner's own writing,
before correction — with `label=0` for every row.

```python
from humanizer_bench.datasets import NonNativeDataset

dataset = NonNativeDataset(split="test", limit=200)
print(len(list(dataset)), "examples")  # construction is cheap: no download yet
```

The `datasets` library comes with the `models` extra
(`pip install -e ".[models]"`); the Hub download happens on first iteration,
and the materialized examples are cached so re-iterating never re-downloads.
The CLI can reach it too: `humanizer-bench --dataset nonnative`.

## FPR per detector

Score every detector on the corpus and compare false-positive rates:

```python
from humanizer_bench.datasets import NonNativeDataset
from humanizer_bench.detectors import HeuristicDetector, PerplexityDetector
from humanizer_bench.metrics import false_positive_rate


def fpr_by_detector(detectors, dataset, threshold=0.5):
    """False-positive rate of each detector on an all-human corpus."""
    examples = list(dataset)
    y_true = [ex.label for ex in examples]
    rows = []
    for det in detectors:
        y_pred = [det.predict(ex.text, threshold) for ex in examples]
        rows.append((det.name, false_positive_rate(y_true, y_pred), len(examples)))
    return rows


dataset = NonNativeDataset(split="test")
print(f"{'detector':<12}{'FPR':>8}{'n':>8}")
print("-" * 28)
for name, fpr, n in fpr_by_detector([HeuristicDetector()], dataset):
    print(f"{name:<12}{fpr:>8.3f}{n:>8}")
```

Expected output (exact numbers vary with the corpus snapshot):

```
detector          FPR       n
----------------------------
heuristic       1.000     747
```

## Reading the result

An FPR of 1.000 looks damning, but read it carefully before citing it. JFLEG
rows are **single sentences**, and `HeuristicDetector` needs at least two
sentences to estimate burstiness — on a single sentence it falls back to the
neutral score 0.5, which at the default threshold 0.5 counts as "AI"
(`predict` uses `>=`). So the toy baseline flags *every* short text, native
or not.

The honest takeaway is therefore not "the heuristic is biased against
learners" but "this baseline cannot judge single-sentence inputs at all" —
which is exactly why the benchmark needs real detectors for a meaningful bias
measurement. Try it: install the `models` extra, swap in `PerplexityDetector`,
and compare its FPR against the heuristic's. A serious detector should keep
FPR low on this corpus; whatever gap remains between corpora is the bias this
project exists to measure.

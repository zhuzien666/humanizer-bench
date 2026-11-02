# Sentence-level false positives: JFLEG learners and matched rewrites

Does this detector score learner writing as more AI-like than a fluent rewrite
of the same sentence? Compare each JFLEG learner sentence with its own human
correction, rather than comparing unrelated corpora whose topic, genre and
length also differ. Both members of a pair have `label=0` (human).

## Matched sample

Use `NonNativeDataset(split="test", limit=600, with_control=True)`. The limit
counts emitted examples, including corrections; it does not mean 600 pairs.
The loader emits each learner sentence immediately followed by the first
nonblank correction that differs from it. A learner without a usable correction
remains unpaired and is excluded from this comparison. These are human fluency
corrections, not independently sampled native-author texts; editing can change
more than grammar, so this comparison does not establish a causal fluency effect.

The measured sample contains **294 matched pairs** among 600 emitted examples.
Keep adjacency and source tags when pairing. Separating all learners and all
corrections and then zipping them can silently shift matches when a correction
is missing; unequal list lengths are an alarm, not a reason to truncate.

## Reproduce

Use Python 3.10 or newer and install the model dependencies:

```bash
python -m pip install -e ".[models]" \
  "transformers==4.57.6" "datasets==4.8.5" "huggingface-hub==0.36.2"
```

The run below used Python 3.10.22, torch 2.14.1, transformers 4.57.6 and datasets
4.8.5 (huggingface-hub 0.36.2), with `PerplexityDetector`'s default `distilbert/distilgpt2` model. Downloads
require access to HuggingFace; model or corpus revisions may change results.
The versions above reproduce the tested environment. The newest Hub client
rejects the existing mixed-corpus loader's unqualified `ag_news` identifier;
that unrelated compatibility issue also reproduces on `main`.

```python
from humanizer_bench.datasets import NonNativeDataset
from humanizer_bench.detectors import PerplexityDetector
from humanizer_bench.metrics import abstention_rate, bootstrap_ci, false_positive_rate

examples = list(NonNativeDataset(split="test", limit=600, with_control=True))
pairs, i = [], 0
while i < len(examples) - 1:
    learner, rewrite = examples[i], examples[i + 1]
    if (not learner.source.endswith(":corrections")
            and rewrite.source.endswith(":corrections")):
        pairs.append((learner.text, rewrite.text))
        i += 2
    else:
        i += 1

if not pairs:
    raise ValueError("No usable matched pairs")

detector = PerplexityDetector()
learner_scores = [detector.score(a) for a, _ in pairs]
rewrite_scores = [detector.score(b) for _, b in pairs]
n = len(pairs)
print(f"pairs {n}")
for name, scores in [("learner", learner_scores), ("rewrite", rewrite_scores)]:
    # Strict > matches BaseDetector.predict(): a neutral 0.5 is not AI.
    fpr = false_positive_rate([0] * n, [int(s > 0.5) for s in scores])
    print(f"{name}: FPR {fpr:.3f}; abstention {abstention_rate(scores):.3f}")

wins = [int(a > b) for a, b in zip(learner_scores, rewrite_scores)]
lo, hi = bootstrap_ci(
    lambda truth, values: sum(values) / len(values),
    [0] * n, wins, n_bootstrap=4000, seed=0,
)
print(f"learner scored more AI-like: {sum(wins) / n:.1%}; "
      f"95% CI [{lo:.3f}, {hi:.3f}]")
```

## Measured result

| Group | Pairs | FPR (score > 0.5) | Abstention (score = 0.5) |
| --- | ---: | ---: | ---: |
| Learner originals | 294 | 0.003 (1/294) | 0.003 (1/294) |
| Matched human rewrites | 294 | 0.024 (7/294) | 0.003 (1/294) |

Learners scored more AI-like than their own rewrites in **12.2%** of pairs
(36/294; paired-example bootstrap 95% CI **[0.088, 0.163]**, 4,000 resamples,
seed 0). There were no score ties. This interval describes the proportion of
pairs with a higher learner score; it is not a confidence interval for the FPR
difference or a subtraction of two marginal intervals.

**The observed direction is opposite to the hypothesis** that this detector
would flag learner sentences more often: the learner FPR is lower, and rewrites
usually receive higher AI scores. This is one detector, one corpus, at sentence
level, on a limited ordered sample. It does not establish how other detectors,
other learner populations, or document-length inputs behave, and a low FPR on
human-only data does not demonstrate useful AI-detection accuracy.

## Why the old heuristic example was misleading

A neutral score of exactly 0.5 means no evidence. The heuristic often returns
that score for single-sentence inputs because it needs multiple sentences to
estimate burstiness. The old example's `FPR = 1.000` depended on a `>=` decision
rule that counted neutral scores as accusations. The current strict `>` rule
avoids that convention artifact. Always report abstention alongside FPR: a
model that cannot judge the input has not demonstrated fairness or accuracy.

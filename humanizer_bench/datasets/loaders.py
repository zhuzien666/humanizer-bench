"""HuggingFace dataset loaders.

Adapts real corpora from the HuggingFace Hub to the :class:`BaseDataset`
interface: one AI-generated corpus plus one human control corpus, so metrics
stop being toy-only. (The non-native-English corpus for false-positive-bias
measurement is a separate Phase-2 issue -- :class:`NonNativeDataset` below.)

Requires the ``models`` extra (``pip install -e '.[models]'``) for the
``datasets`` library. The Hub import is deferred to first iteration and the
materialized examples are cached on the instance, so iterating stays
repeatable and never re-downloads.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence

from .._deps import require
from .base import BaseDataset, Example
from typing import Iterator, List


class HFDataset(BaseDataset):
    """A single HuggingFace Hub corpus mapped onto :class:`Example` records.

    Every row gets the same ``label`` -- pair an AI corpus (``label=1``) with
    a human corpus (``label=0``) via :func:`load_hf_dataset` for a mixed
    benchmark set.

    Args:
        hf_name: Dataset repo on the Hub, e.g. ``"ag_news"``.
        label: Fixed label for every example: ``1`` = AI, ``0`` = human.
        split: Hub split to load, e.g. ``"train"``.
        limit: Maximum examples to keep (first rows); ``None`` keeps all.
        text_field: Column holding the document text. Rows with a missing or
            blank value in this column are skipped.
        source: Provenance tag on each example; defaults to ``hf_name``.
    """

    name = "hf"

    def __init__(
        self,
        hf_name: str,
        label: int = 0,
        split: str = "train",
        limit: int | None = None,
        text_field: str = "text",
        source: str | None = None,
    ) -> None:
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit!r}")
        self.hf_name = hf_name
        self.label = label
        self.split = split
        self.limit = limit
        self.text_field = text_field
        self.source = source if source is not None else hf_name
        self._examples: list[Example] | None = None

    def _load(self) -> list[Example]:
        """Download the split and materialize examples, once."""
        if self._examples is not None:
            return self._examples

        datasets = require("datasets")
        ds = datasets.load_dataset(self.hf_name, split=self.split)
        if self.text_field not in ds.column_names:
            raise ValueError(
                f"column {self.text_field!r} not in {self.hf_name!r} "
                f"(available: {', '.join(ds.column_names)})"
            )

        examples: list[Example] = []
        for row in ds:
            text = row.get(self.text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            examples.append(
                Example(text=text, label=self.label, source=self.source)
            )
            if self.limit is not None and len(examples) >= self.limit:
                break
        self._examples = examples
        return examples

    def __iter__(self) -> Iterator[Example]:
        return iter(self._load())


class _CombinedDataset(BaseDataset):
    """Concatenate several datasets, preserving order.

    Iteration is repeatable because each wrapped dataset is repeatable.
    """

    name = "hf-mixed"

    def __init__(self, datasets: Sequence[BaseDataset]) -> None:
        self._datasets = list(datasets)

    def __iter__(self) -> Iterator[Example]:
        for dataset in self._datasets:
            yield from dataset


def load_hf_dataset(
    ai_dataset: str = "tatsu-lab/alpaca",
    human_dataset: str = "ag_news",
    *,
    split: str = "train",
    limit: int | None = None,
    ai_text_field: str = "output",
    human_text_field: str = "text",
) -> BaseDataset:
    """Load one AI-generated corpus and one human corpus from the HF Hub.

    Returns a :class:`BaseDataset` yielding AI examples (``label=1``) from
    ``ai_dataset`` followed by human examples (``label=0``) from
    ``human_dataset``, each tagged with its corpus as ``source``. Iteration is
    repeatable; pass ``limit`` to cap examples *per corpus* and keep runs fast.

    The defaults pair GPT-3-generated instructions (``tatsu-lab/alpaca``)
    with human-written news (``ag_news``) -- both public, no auth required.

    Args:
        ai_dataset: Hub repo for the AI-generated corpus.
        human_dataset: Hub repo for the human-written corpus.
        split: Hub split to load from both corpora.
        limit: Maximum examples per corpus; ``None`` keeps everything.
        ai_text_field: Text column in the AI corpus.
        human_text_field: Text column in the human corpus.
    """
    return _CombinedDataset(
        [
            HFDataset(
                ai_dataset,
                label=1,
                split=split,
                limit=limit,
                text_field=ai_text_field,
            ),
            HFDataset(
                human_dataset,
                label=0,
                split=split,
                limit=limit,
                text_field=human_text_field,
            ),
        ]
    )


class HFMixedDataset(BaseDataset):
    """HuggingFace benchmark set: one AI corpus + one human corpus.

    Constructible with no arguments so it can live in the component registry --
    ``humanizer-bench --dataset hf`` resolves to this. The corpora are
    :func:`load_hf_dataset`'s defaults (GPT-3-generated Alpaca instructions
    as the AI class, human-written AG News as the human class); call
    :func:`load_hf_dataset` directly for custom corpora.

    Construction stays cheap -- nothing is downloaded until first iteration --
    but *iteration* is not: the two corpora hold 172k rows between them, and a
    detector pass is ~31 ms/text (back-translation ~1.1 s/text), so an uncapped
    run takes hours. Hence the small default ``limit``; raise it deliberately.

    Args:
        limit: Maximum examples per corpus. The default of 50 yields 100
            examples: roughly 12 s for a detector-only matrix, 70 s including
            back-translation.
    """

    name = "hf"

    def __init__(self, limit: int = 50) -> None:
        self._inner = load_hf_dataset(limit=limit)

    def __iter__(self) -> Iterator[Example]:
        return iter(self._inner)


class NonNativeDataset(BaseDataset):
    """Non-native English writing from the JFLEG corpus, all human-labeled.

    Every example comes from the ``sentence`` field of ``jhu-clsp/jfleg`` --
    sentences written by English-language learners with different first
    languages (Napoles et al., 2017). All rows are human writing, so every
    example gets ``label=0``. Built for the false-positive experiment: run
    detectors over it and compute ``false_positive_rate`` to see how often
    real human writers get flagged as AI.

    The Hub import is deferred to first iteration and the materialized
    examples are cached on the instance, so constructing the dataset is cheap
    (no download) and iterating stays repeatable without re-downloading.

    Set ``with_control=True`` to pair every learner sentence with the first
    of JFLEG's four human corrections of that same sentence. Both sides are
    human writing (``label=0``) with the same content and the same median
    length, so a gap between them isolates fluency -- which comparing JFLEG
    against a different corpus cannot do, because topic, genre and length all
    move at once.

    Args:
        split: Hub split to load (``"test"`` or ``"validation"``).
        limit: Maximum examples to keep (first rows); ``None`` keeps all.
            Counts the control rewrites too when ``with_control`` is set.
        with_control: Also emit the matched native rewrite of each sentence.
    """

    name = "nonnative"

    #: Hub repo holding the corpus.
    hf_name = "jhu-clsp/jfleg"

    #: Splits the JFLEG repo actually publishes (there is no "train").
    valid_splits = frozenset({"test", "validation"})

    def __init__(
        self,
        split: str = "test",
        limit: int | None = None,
        with_control: bool = False,
    ) -> None:
        if split not in self.valid_splits:
            raise ValueError(
                f"split must be one of {sorted(self.valid_splits)}, got {split!r}"
            )
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit!r}")
        self.split = split
        self.limit = limit
        self.with_control = with_control
        self._examples: List[Example] | None = None

    def __iter__(self) -> Iterator[Example]:
        if self._examples is None:
            self._examples = self._load()
        return iter(self._examples)

    def _load(self) -> List[Example]:
        datasets = require("datasets")
        hf = datasets.load_dataset(self.hf_name, split=self.split)
        examples: List[Example] = []
        for row in hf:
            text = (row.get("sentence") or "").strip()
            if not text:
                continue
            examples.append(Example(text=text, label=0, source=self.hf_name))
            # Emit the matched native rewrite right after each learner
            # sentence. Both are human (label 0); they differ only in
            # fluency, the one variable the experiment is about.
            if self.with_control:
                control = self._first_correction(row, text)
                if control is not None:
                    examples.append(
                        Example(
                            text=control,
                            label=0,
                            source=f"{self.hf_name}:corrections",
                        )
                    )
            if self.limit is not None and len(examples) >= self.limit:
                break
        return examples

    @staticmethod
    def _first_correction(row: dict, sentence: str) -> str | None:
        """First fluent rewrite of `sentence` that actually differs from it.

        JFLEG ships four human corrections per row. Rewrites identical to the
        original carry no contrast, so they are skipped.
        """
        for candidate in row.get("corrections") or []:
            cleaned = (candidate or "").strip()
            if cleaned and cleaned != sentence:
                return cleaned
        return None


def load_nonnative_dataset(
    split: str = "test", limit: int | None = None, with_control: bool = False
) -> BaseDataset:
    """Load the JFLEG non-native-English corpus as a :class:`BaseDataset`.

    Convenience wrapper around :class:`NonNativeDataset`.
    """
    return NonNativeDataset(split=split, limit=limit, with_control=with_control)

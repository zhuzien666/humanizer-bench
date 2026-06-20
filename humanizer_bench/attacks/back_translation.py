"""Back-translation attack (Phase 2 -- not yet implemented).

The real implementation will translate ``EN -> pivot language -> EN`` using
MarianMT models (Helsinki-NLP) so the round-trip paraphrases the text. Kept as a
stub for now so the import path is stable.
"""

from __future__ import annotations

from .base import BaseAttack


class BackTranslationAttack(BaseAttack):
    """Placeholder for a MarianMT round-trip paraphrase attack."""

    name = "back_translation"

    def __init__(self, pivot: str = "fr") -> None:
        self.pivot = pivot

    def transform(self, text: str) -> str:  # pragma: no cover - stub
        raise NotImplementedError(
            "BackTranslationAttack is a Phase 2 stub. Wire up Helsinki-NLP "
            "MarianMT (EN->pivot->EN). Install extras with "
            "`pip install -e '.[models]'`."
        )

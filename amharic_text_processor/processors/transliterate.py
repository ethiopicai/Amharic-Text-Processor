"""Transliteration processor for Ethiopic (Amharic) text."""

from __future__ import annotations

from typing import Dict

from amharic_text_processor.assets.romanization import ROMANIZATION
from amharic_text_processor.base import BaseProcessor, ProcessorInput, ProcessorOutput


class AmharicTransliterator(BaseProcessor):
    """Transliterate Ethiopic (Amharic) text to Latin script using a romanization table."""

    def __init__(self, mapping: Dict[str, str] | None = None) -> None:
        self.mapping = dict(ROMANIZATION if mapping is None else mapping)

    def apply(self, data: ProcessorInput) -> ProcessorOutput:
        text = BaseProcessor._extract_text(data)
        transliterated = "".join(self.mapping.get(ch, ch) for ch in text)
        return {"text": transliterated, "transliteration_applied": transliterated != text}

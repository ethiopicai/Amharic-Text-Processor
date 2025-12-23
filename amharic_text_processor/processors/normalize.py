"""Normalization processors."""

from __future__ import annotations

import re
import unicodedata

from amharic_text_processor.base import BaseProcessor, ProcessorInput, ProcessorOutput


class PunctuationNormalizer:
    """Unify punctuation characters and reduce repeats."""

    PUNCT_TRANSLATION = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "、": ",",
            "；": ";",
            "：": ":",
        }
    )

    def apply(self, data: ProcessorInput) -> ProcessorOutput:
        text = BaseProcessor._extract_text(data)
        # Protect decimal numbers so spacing rules do not split them.
        decimals: list[tuple[str, str]] = []

        def _capture_decimal(match: re.Match[str]) -> str:
            token = f"__DECIMAL_{len(decimals)}__"
            decimals.append((token, match.group(0)))
            return token

        protected = re.sub(r"\d+\.\d+", _capture_decimal, text)

        normalized = protected.translate(self.PUNCT_TRANSLATION)
        punct_all = r"[?!.,;:።፣፤፥፦፧፨]"
        punct_spacing = r"[?!,;:።፣፤፥፦፧፨]"
        # Collapse repeated punctuation to a single mark.
        normalized = re.sub(rf"({punct_all}){{2,}}", r"\1", normalized)
        # Do not insert spaces after punctuation (to preserve formats like 8.5%).
        # Restore protected decimals intact.
        for token, value in decimals:
            normalized = normalized.replace(token, value)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return {"text": normalized, "punctuation_normalized": normalized != text}


class UnicodeNormalizer:
    """Normalize Unicode to a specific form (default NFC) and optionally strip control characters."""

    def __init__(self, form: str = "NFC", strip_control: bool = True) -> None:
        self.form = form
        self.strip_control = strip_control

    def apply(self, data: ProcessorInput) -> ProcessorOutput:
        text = BaseProcessor._extract_text(data)
        normalized = unicodedata.normalize(self.form, text)
        if self.strip_control:
            normalized = "".join(ch for ch in normalized if unicodedata.category(ch)[0] != "C")
        return {"text": normalized, "unicode_normalized": normalized != text}


class CharacterRemapper:
    """Remap legacy/variant Ethiopic characters to canonical forms."""

    # Character-level canonicalization for commonly interchanged Ethiopic glyphs.
    REMAP = {
        "ሠ": "ሰ",
        "ሡ": "ሱ",
        "ሢ": "ሲ",
        "ሣ": "ሳ",
        "ሤ": "ሴ",
        "ሥ": "ስ",
        "ሦ": "ሶ",
        "ሧ": "ሷ",
        "ሐ": "ሀ",
        "ሑ": "ሁ",
        "ሒ": "ሂ",
        "ሓ": "ሀ",
        "ሔ": "ሄ",
        "ሕ": "ህ",
        "ሖ": "ሆ",
        "ሃ": "ሀ",
        "ኀ": "ሀ",
        "ኁ": "ሁ",
        "ኂ": "ሂ",
        "ኃ": "ሀ",
        "ኄ": "ሄ",
        "ኅ": "ህ",
        "ኆ": "ሆ",
        "ኊ": "ሂ",
        "ኋ": "ሀ",
        "ኌ": "ሄ",
        "ኍ": "ህ",
        "ኾ": "ሆ",
        "ፀ": "ጸ",
        "ፁ": "ጹ",
        "ፂ": "ጺ",
        "ፃ": "ጻ",
        "ፄ": "ጼ",
        "ፅ": "ጽ",
        "ፆ": "ጾ",
        "ፇ": "ጿ",
        "ዯ": "ዮ",
        "ዐ": "አ",
        "ዑ": "ኡ",
        "ዒ": "ኢ",
        "ዓ": "አ",
        "ዔ": "ኤ",
        "ዕ": "እ",
        "ዖ": "ኦ",
        "ጎ": "ጐ",
        "ኰ": "ኮ",
        "ቊ": "ቁ",
        "ኵ": "ኩ",
        "ዉ": "ው",
    }

    def __init__(self) -> None:
        self._translation_table = str.maketrans(self.REMAP)

    def apply(self, data: ProcessorInput) -> ProcessorOutput:
        text = BaseProcessor._extract_text(data)
        remapped = text.translate(self._translation_table)
        # Normalize labialized suffix variants like ቱዋል -> ቷል.
        labialized = {
            "ሉ": "ሏ",
            "ሙ": "ሟ",
            "ቱ": "ቷ",
            "ሩ": "ሯ",
            "ሱ": "ሷ",
            "ሹ": "ሿ",
            "ቁ": "ቋ",
            "ቡ": "ቧ",
            "ቹ": "ቿ",
            "ሁ": "ኋ",
            "ኑ": "ኗ",
            "ኙ": "ኟ",
            "ኩ": "ኳ",
            "ዙ": "ዟ",
            "ጉ": "ጓ",
            "ደ": "ዷ",
            "ጡ": "ጧ",
            "ጩ": "ጯ",
            "ጹ": "ጿ",
            "ፉ": "ፏ",
        }
        for base, normalized_form in labialized.items():
            remapped = re.sub(rf"({base}[ዋአ])", normalized_form, remapped)
        return {"text": remapped, "characters_remapped": remapped != text}

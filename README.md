# Amharic Text Processor
[![PyPI](https://img.shields.io/pypi/v/amharic-text-processor.svg)](https://pypi.org/project/amharic-text-processor/) [![CI](https://github.com/isrish/Amharic-Text-Processor/actions/workflows/test.yml/badge.svg)](https://github.com/isrish/Amharic-Text-Processor/actions/workflows/test.yml) [![Docs](https://github.com/isrish/Amharic-Text-Processor/actions/workflows/publish.yml/badge.svg?label=docs)](https://github.com/isrish/Amharic-Text-Processor/actions/workflows/publish.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Amharic Text Processor** is a modular Python toolkit for cleaning, normalizing, and formatting Amharic text. Each processing step is a small class with a predictable `.apply()` method, and steps are easily chained with `Pipeline`.

**Why this exists:** Amharic text from the web, documents, and OCR often arrives with HTML noise, mixed Ethiopic variants, inconsistent punctuation, legacy abbreviations, and numerals in different forms. This toolkit provides predictable, composable processors so you can rapidly build robust pipelines for ML datasets, search indexing, or downstream NLP tasks without reinventing cleaning logic. Many of these components were developed while processing large volumes of Amharic text crawled from Amharic-focused websites indexed by [Common Crawl](https://commoncrawl.org/get-started).

---

## ✨ Features

- Composable pipeline built from simple processor classes
- Consistent I/O contract: accepts `str` or `{"text": ...}`, returns a dict with `"text"`
- HTML stripping, whitespace cleanup, Amharic character filtering
- Punctuation and Unicode normalization (keeps Ethiopic marks, preserves decimals) plus configurable regex filtering
- Sentence-level deduplication using fuzzy similarity
- Abbreviation handling for slash/dot forms; dotted abbreviations can be normalized before expansion
- Helpers to add spaces between Ethiopic letters and digits, and to place sentences on separate lines
- Noise removal for common Latin/underscore tokens and foreign-only brackets
- Ethiopic→Latin transliteration using a romanization table
- Pure, side-effect-free processors that are easy to test and extend

---

## 📦 Installation

```bash
pip install amharic-text-processor
```

---

## 🚀 Quick Start

```python
from amharic_text_processor import Pipeline
from amharic_text_processor.processors import (
    HtmlStripper,
    WhitespaceNormalizer,
    PunctuationNormalizer,
    UnicodeNormalizer,
    CharacterRemapper,
    AbbreviationExpander,
    DottedAbbreviationNormalizer,
    AmharicCharacterFilter,
    CommonNoiseRemover,
)

pipeline = Pipeline([
    HtmlStripper(),             # drop HTML/script/style
    UnicodeNormalizer(),        # NFC + strip control chars
    CharacterRemapper(),        # normalize Ethiopic variants (ሠ->ሰ, ዐ->አ, ...)
    DottedAbbreviationNormalizer(),  # turn dotted abbreviations into slash form
    AbbreviationExpander(),     # expand slash/dot abbreviations (e.g., ዓ.ም. -> ዓመተ ምሕረት)
    PunctuationNormalizer(),    # unify punctuation (keeps Ethiopic marks, protects decimals)
    WhitespaceNormalizer(),     # collapse repeated whitespace
    AmharicCharacterFilter(),   # keep Ethiopic chars and safe punctuation/digits
    CommonNoiseRemover(),       # drop tokens like IMG_1124 or (FlyDubai)
])

raw = """
<article>
  <p>  ሰላም። ልኡኣ ዓ.ም. 2016 ሀ/ማርያም በሚሊዮን ይዘት ሰጠ። </p>
  <script>alert('ignore me')</script>
</article>
"""

result = pipeline.apply(raw)
print(result["text"])
# -> ሰላም። ሏ ዓመተ ምሕረት 2016 ሀይለ ማርያም በሚሊዮን ይዘት ሰጠ።

# Transliteration to Latin
rawtext = "እሺ፣ የክፍያ ሂደቱን በአማርኛ እመራዎታለሁ። የአካውንት ቁጥርዎን ያስገቡ። 565 የኢትዮጵያ ንግድ ባንክ ነው።"
new_text = AmharicTransliterator().apply(rawtext)
print(new_text["text"])
# -> eshi, yakefeyaa hidatune baamaarenyaa emaraawotaalahu. yaakaawenete quterewone yaasegabu. 565 yaiteyopheyaa negede baaneke nawe.
```

---

## 🔗 Pipeline Contract

- Input: `str` or `dict` containing `"text": str`
- Output: always a `dict` with at least `"text": str`
- Processors run in order; output from one is passed to the next
- Fail-fast validation on invalid inputs or processor outputs

## 📚 Code Documentation

- Each processor and the pipeline include docstrings describing inputs/outputs and behavior (see `amharic_text_processor/base.py`, `pipeline.py`, and files in `amharic_text_processor/processors/`).
- Browse in an editor or via `pydoc amharic_text_processor.processors.<name>` for details.
- All processors follow the same contract: `.apply(data: str | {"text": str}) -> {"text": str, ...}`.
- See `docs/` for a quick reference (`docs/index.md`, `docs/processors.md`). To generate HTML docs locally you can run `pdoc -o docs amharic_text_processor`.

---

## 🧰 Built-in Processors

- [`HtmlStripper`](amharic_text_processor/processors/html.py): remove HTML tags and script/style content
- [`WhitespaceNormalizer`](amharic_text_processor/processors/whitespace.py): collapse repeated whitespace and trim
- [`PunctuationNormalizer`](amharic_text_processor/processors/normalize.py): unify Ethiopic/ASCII punctuation, collapse repeats, keep decimals intact
- [`UnicodeNormalizer`](amharic_text_processor/processors/normalize.py): normalize Unicode (default NFC) and strip control chars
- [`AmharicCharacterFilter`](amharic_text_processor/processors/filters.py): keep Ethiopic characters plus safe punctuation/digits
- [`CharacterRemapper`](amharic_text_processor/processors/normalize.py): normalize variant Ethiopic glyphs to canonical forms
- [`DottedAbbreviationNormalizer`](amharic_text_processor/processors/abbreviations.py): convert dotted abbreviations (e.g., እ.ኤ.አ) into slash form before expansion
- [`AbbreviationExpander`](amharic_text_processor/processors/abbreviations.py): expand slash/dot Amharic abbreviations to full forms (e.g., ፍ/ቤቱ -> ፍርድ ቤቱ, ፕ/ር -> ፕሮፌሰር, ዓ.ም. -> ዓመተ ምሕረት)
- [`NumberToGeez`](amharic_text_processor/processors/numbers.py): convert Arabic digits in text to Ethiopic (Geez) numerals (e.g., 31 -> ፴፩)
- [`GeezToNumber`](amharic_text_processor/processors/numbers.py): convert Ethiopic (Geez) numerals back to Arabic digits  (e.g., ፴፩ -> 31)
- [`WordNumberToDigits`](amharic_text_processor/processors/numbers.py): convert Amharic worded numbers (e.g., “ሁለት ሺህ ሶስት መቶ”) to Arabic digits, including millions+
- [`DigitsToWordNumber`](amharic_text_processor/processors/numbers.py): turn Arabic digit sequences into Amharic worded numbers (supports up to trillions)
- [`OldPhoneMapper`](amharic_text_processor/processors/phonetic.py): convert legacy phone representations to modern forms via a predefined mapping
- [`EthiopicNumberSpacer`](amharic_text_processor/processors/tokenize.py): insert spaces between Ethiopic letters and adjacent digits (e.g., "ዜና11" -> "ዜና 11")
- [`SentenceLineFormatter`](amharic_text_processor/processors/tokenize.py): place each sentence on its own line after end punctuation
- [`SentenceDeduplicator`](amharic_text_processor/processors/deduplication.py): drop exact or near-duplicate sentences with RapidFuzz similarity
- [`AmharicTransliterator`](amharic_text_processor/processors/transliterate.py): transliterate Ethiopic (Amharic) text to Latin using a romanization table
- [`CommonNoiseRemover`](amharic_text_processor/processors/filters.py): remove noisy tokens like `IMG_1124` or non-Ethiopic bracketed text `(some_not_amharic_words)`
- [`RegexFilter`](amharic_text_processor/processors/filters.py): run a configurable regex substitution with counts

### Sentence deduplication example

```python
from amharic_text_processor.processors import SentenceDeduplicator

deduper = SentenceDeduplicator(similarity_threshold=0.85)
text = "ሰላም ዓለም። ሰላም ዓለም። እንዴት ነህ? እርስዎ እንዴት ነው?"
result = deduper.apply(text)
print(result["text"])
# -> ሰላም ዓለም። እንዴት ነህ?
print(result["sentences_removed"])  # duplicates that were dropped
```

---

## 🧧 Custom Processor Example

```python
from amharic_text_processor import BaseProcessor


class ExampleProcessor(BaseProcessor):
    def apply(self, data):
        text = BaseProcessor._extract_text(data)
        processed = text.replace("old", "new")
        return {"text": processed, "modified": True}
```

Add it to a pipeline just like the built-ins.

---

## 🧪 Testing

```bash
pytest -q
```

## 🤝 Contributing

See CONTRIBUTING.md for guidelines on adding processors, running tests, and coding style.

## 📦 Publishing

GitHub Actions workflows are included:
- `CI` runs tests on pushes/PRs.
- `Publish to PyPI` builds and publishes on release creation.
- See CHANGELOG.md for release notes.

---

## 📜 License

MIT License.

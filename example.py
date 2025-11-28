from amharic_text_processor import Pipeline
from amharic_text_processor.processors import (
    HtmlStripper,
    WhitespaceNormalizer,
    PunctuationNormalizer,
    UnicodeNormalizer,
    AmharicCharacterFilter,
    AbbreviationExpander,
    NumberToGeez,
    GeezToNumber,
    DigitsToWordNumber,
    WordNumberToDigits,
    SentenceDeduplicator,
    SentenceLineFormatter,
    DottedAbbreviationNormalizer,
    CommonNoiseRemover,
    AmharicTransliterator,
)

pipeline = Pipeline([
    HtmlStripper(),
    CommonNoiseRemover(),
    WhitespaceNormalizer(),
    PunctuationNormalizer(),
    UnicodeNormalizer(strip_control=False),
    AmharicCharacterFilter(),
    DottedAbbreviationNormalizer(),
    AbbreviationExpander(),
    NumberToGeez(),
    GeezToNumber(),
    SentenceDeduplicator(),
    WhitespaceNormalizer(),
    SentenceLineFormatter(), # Add new line after sentences. optional used for output formatting.
])

raw = "<p>  የግሎባል... ሰላም  እንዴት  ነህ። ካፒታሊዝም የሚያበቃለት ይመስለኛል ሲሉ ረዳት ፕ/ር መድኃኔ ታደሰ ለሩሕ በሰጡት ቃለ ምልልስ ገለፁ።</p>"
result = pipeline.apply(raw)
print(raw)
print(result["text"])  # -> ሰላም እንዴት ነህ.

new_text = AbbreviationExpander().apply("ዶ/ር አብ/ር እና ፕ/ር መድኃኔ")
print(new_text["text"])  # -> ዶክተር አብርሃም እና ፕሮፌሰር መድኃኔ  

raw = "ሰላሳ አንድ ፴፩ 31 ከፍተኛ ቤት ነው። 565 የኢትዮጵያ ንግድ ባንክ ነው።"
new_text = NumberToGeez().apply(raw)
print(new_text["text"])  # -> ሰላሳ አንድ ፴፩ ፴፩ ከፍተኛ ቤት ነው። ፭፻፷፭ የኢትዮጵያ ንግድ ባንክ ነው።

raw = "ሰላሳ አንድ ፴፩ 31 ከፍተኛ ቤት ነው። 565 የኢትዮጵያ ንግድ ባንክ ነው።"
new_text = GeezToNumber().apply(raw)
print(new_text["text"])  # -> ሰላሳ አንድ 31 31 ከፍተኛ ቤት ነው።

raw = "በሶማሌ ክልል በቀብሪደሃር ዞን 543 ሰዎች ከሽፍቶች ጋር ግንኙነት አላችሁ። እነሱም 24 ወንዶች እና 123,999,888 ሴቶች ናቸው።"
new_text = DigitsToWordNumber().apply(raw)
print(new_text["text"])  # -> በሶማሌ ክልል በቀብሪደሃር ዞን አምስት መቶ አርባ ሶስት ሰዎች ከሽፍቶች ጋር ግንኙነት አላችሁ። እነሱም ሃያ አራት ወንዶች እና መቶ ሃያ ሶስት ሴቶች ናቸው።

# this is a complex example combining several number words. rarely used in practice but good for testing
raw = "በ2017 ዓ.ም. ከጥራጥሬ ምርት 272.5 ሚሊዮን ዶላር፣ ከሰሊጥ ምርት ደግሞ 334 ሚሊዮን ዶላር መገኘቱን፣ ለዚህ ገቢ መገኘት ላኪዎች ከጥራት አኳያ ማሟላት ያለባቸውን መሥፈርት እየተረዱ መሆኑን"
new_text = WordNumberToDigits().apply(raw)
print(new_text["text"])  # 

raw = "በ2017 ዓ.ም. ከጥራጥሬ ምርት 272.5 ሚሊዮን ዶላር፣ ከሰሊጥ ምርት ደግሞ 334 ሚሊዮን ዶላር መገኘቱን፣ ለዚህ ገቢ መገኘት ላኪዎች ከጥራት አኳያ ማሟላት ያለባቸውን መሥፈርት እየተረዱ መሆኑን"
new_text = pipeline.apply(raw)
print(new_text["text"])  # 
# . -> 

print("----- Example of full pipeline on sample_crawled.txt -----")
with open("sample_crawled.txt", "r", encoding="utf-8") as f:
    raw = f.read()
    cleaned = pipeline.apply(raw)
    print(cleaned["text"][:500])  # print first 500 characters of cleaned text
    with open("cleaned_output.txt", "w", encoding="utf-8") as out_f:
        out_f.write(cleaned["text"])  # save cleaned text to file


# transliterate Amharic text to Latin using a romanization table
rawtext = "እሺ፣ የክፍያ ሂደቱን በአማርኛ እመራዎታለሁ። የአካውንት ቁጥርዎን ያስገቡ። 565 የኢትዮጵያ ንግድ ባንክ ነው።"
new_text = AmharicTransliterator().apply(rawtext)
print(new_text["text"])  # 
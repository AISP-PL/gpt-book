"""
This module contains the parsers for the GPT-3 book.
"""


def autosplit_paragraphs(text: str, max_words: int = 400) -> list[str]:
    """
    Split long text into multiple paragraphs based on max words inside paragraph.
    All senteces must be fitted together inside paragraph.
    """
    # Paragraphs : Split
    paragraphs = text.split("\n")

    # Sentences : Split
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(paragraph.split(". "))

    # Sentences : Change to tuples (text, word count)
    sentences = [(sentence.strip(), len(sentence.split(" "))) for sentence in sentences]

    # Fix: If single sentence is longer than max_words, then split it
    # @TODO

    # Paragraphs : Create from sentences
    paragraphs = []
    paragraph, paragraph_words = "", 0
    for sentence, word_count in sentences:
        # Paragraph : Add sentence
        paragraph += sentence + ". "
        paragraph_words += word_count

        # Check if paragraph is full
        if paragraph_words > max_words:
            paragraphs.append(paragraph)
            paragraph, paragraph_words = "", 0

    # Last paragraph : Add
    if paragraph_words > 0:
        paragraphs.append(paragraph)

    return paragraphs

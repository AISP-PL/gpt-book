"""
    OpenAI GPT-3 Book Writer Agent class (dataclass), which :
    - has a system prompt in Polish that he is a book writer/redactor from unstructured text,
    - process() method inputs a text file as string, and outputs a list of paragraphs pairs (original, generated),
    - process() processing is done in a loop for each paragraph, and the cost of each API call is added to the billings
    - process() loop uses current original paragraph and last generated paragraph as input for the next API call,
"""

from dataclasses import dataclass

import openai
from openai import OpenAI

from gpt_book.helpers.parsers import autosplit_paragraphs
from gpt_book.models.billings import Billings


@dataclass
class BookWriter:
    """Class for book writer agent."""

    def process(
        self,
        client: OpenAI,
        text: str,
        model: str,
        model_tokens: int,
        billings: Billings,
        paragraph_max_words: int = 200,
    ) -> list[tuple[str, str]]:
        """
        Process input text with OpenAI model and return list
        of paragraphs pairs (original, generated).
        """
        # Split text into paragraphs
        input_paragraphs = autosplit_paragraphs(text, max_words=paragraph_max_words)

        # Process each paragraph
        output_paragraphs: list[tuple[str, str]] = []

        # Max tokens : Calculate
        max_generated_tokens = max(500, model_tokens - 1000)

        # Last generated paragraph
        last_generated = ""
        # Process : Loop
        for paragraph in input_paragraphs:
            # Process : Generate
            generated = (
                client.completions.create(
                    model=model,
                    prompt=f"Jesteś redaktorem książki którą tworzysz na podstawie transkrypcji wykładu mówionego. \
                             Zredaguj poniższy framgent tekstu z zapisu audio bezpośrednio jako fragment książki. \
                             Użyj współczesnego języka polskie, książkowego, literackiego. \
                             Usuń niepotrzebne chrząknięcia czy fragmenty gdy autor zastanawiał się nad myślą. \n\n \
                             Oto fragment książki poprzedzający to co będziesz redagować : {last_generated} \n\n \
                             Oto oryginalny tekst zapisu audio:\n\n{paragraph}\n\n\
                             Oto twój tekst :\n\n",
                )
                .choices[0]
                .text.strip()
            )

            # Add cost to billings
            billings.add_api_call(model)

            # Add to output
            output_paragraphs.append((paragraph, generated))
            last_generated = generated

        return output_paragraphs

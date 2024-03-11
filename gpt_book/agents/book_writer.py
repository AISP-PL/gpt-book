"""
    OpenAI GPT-3 Book Writer Agent class (dataclass), which :
    - has a system prompt in Polish that he is a book writer/redactor from unstructured text,
    - process() method inputs a text file as string, and outputs a list of paragraphs pairs (original, generated),
    - process() processing is done in a loop for each paragraph, and the cost of each API call is added to the billings
    - process() loop uses current original paragraph and last generated paragraph as input for the next API call,
"""

from copy import copy
from dataclasses import dataclass

import openai
from gradio import Progress
from openai import OpenAI
from tqdm import tqdm

from gpt_book.helpers.chat_gpt import GptPrompt
from gpt_book.helpers.parsers import autosplit_paragraphs
from gpt_book.models.billings import Billings
from gpt_book.models.text_comparison import TextComparison


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
    ) -> list[TextComparison]:
        """
        Process input text with OpenAI model and return list
        of paragraphs pairs (original, generated).
        """
        progress = Progress()

        # Split text into paragraphs
        input_paragraphs = autosplit_paragraphs(text, max_words=paragraph_max_words)

        # Process each paragraph
        output_paragraphs: list[tuple[str, str]] = []

        # Max tokens : Calculate
        max_generated_tokens = max(500, model_tokens - 1000)

        # Last generated paragraph
        last_generated = ""
        # Process : Loop
        for paragraph in progress.tqdm(
            input_paragraphs, desc="Processing book paragraphs"
        ):
            # OpenAI : Messages
            messages = [
                {
                    "role": "system",
                    "content": "Jesteś redaktorem książki którą tworzysz na podstawie zapisu wykładu mówionego. \
                                    Używasz prostych i celnych zdań, literackiego języka polskiego. \
                                    Usuwasz niepotrzebne zdania lub fragmenty 'nie wiem', 'myślę', gdy autor się zastanawiał.\
                                    Zmieniasz fragmenty zdań, gdy autor nie jest pewny siebie lub poprawisz błędy merytoryczne, zgodnie z przyjęta wiedzą. \
                                    Usuwasz zdania bez czasowników lub bez myśli przewodniej. \
                                    Usuwasz błędy językowe, interpunkcyjne, stylistyczne.\
                                    Uzupełniasz brakujące słowa, zdania oraz fragmenty zdań.\
                                    Emocje to radość, smutek, lęk oraz gniew. Nie nazywaj emocjami innych rzeczy jak np. miłość.\
                                    ",
                },
                {
                    "role": "user",
                    "content": f"Oto oryginalny tekst zapisu audio, który wymaga redakcji : {paragraph}\n\n",
                },
                {
                    "role": "user",
                    "content": f"Dla kontekstu podaje Ci fragment książki, który poprzedza to co będziesz redagować. : {last_generated}.\n\n\
                                 Nie powtarzaj tekstu kontekstu, ale pisz tylko na bazie oryginalnego zapisu audio. Pisz.",
                },
            ]

            # Process : Generate
            generated_paragraph = None
            for try_index in range(3):
                # Prompt : Generate
                generated_paragraph = GptPrompt(client, messages, model=model)

                # Check : Generated valid paragraph
                if generated_paragraph is not None:
                    break

                # Check : Last try
                if try_index == 2:
                    raise ValueError(
                        "OpenAI GPT : Connection error! Failed to generate paragraph!"
                    )

            # Add cost to billings
            billings.add_api_call(model)

            # Output : Append
            output_paragraphs.append(
                TextComparison(input_text=paragraph, output_text=generated_paragraph)
            )

            # Last generated : Save
            last_generated = copy(generated_paragraph)

        return output_paragraphs

"""
    Wrapper around openai functions, to handle :
    - prompt saving,
    - buffering prompts,
    - chat conversation handling in cyclic buffer,
"""

import logging
from typing import Optional

from openai import OpenAI


def GptPrompt(
    client: OpenAI,
    messages: list,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
) -> Optional[str]:
    """Simple wrapper prompt GPT with messages."""

    # Send request to Open AI
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )

    # Check : Invalid response
    if response is None:
        logging.error("Open AI response is invalid!")
        return None

    # Check : Choices empty
    if len(response.choices) == 0:
        logging.error("Open AI response is empty!")
        return None

    # Choice : Get first
    response_choice = response.choices[0]

    # Messsage : Get content from response
    message = response_choice.message

    # Check : Message empty
    if (message is None) or (len(message.content) == 0):
        logging.error("Open AI response message is empty!")
        return None

    return message.content

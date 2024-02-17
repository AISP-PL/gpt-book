"""
 Pytest test of parser.
"""

import pytest

from gpt_book.helpers.parsers import autosplit_paragraphs


def test_paragraph():
    """Test"""

    text = "This is a test. This is a test. This is a test. Walnuts are good for health \
    and brain. Walnuts are good for health and brain. Walnuts are good for health and brain because they contain omega-3 fatty acids. \
    Every pokemon has a unique ability for training. If you want to be a pokemon master, you have to train your pokemon. \
    Combine walnuts and pokemon to get a healthy brain and body. But, don't forget to train your pokemon. \
    Pikachu is the most famous pokemon, and not Charmander. Charmander is also a good pokemon, but Pikachu is the best. \
    Pikachu is the best pokemon because it has the ability to generate electricity. Charmander is a fire pokemon. \
    Fire pokemon are good for cooking. Cooking is an art. Art is a way to express yourself. \
    Express yourself with cooking and art. Art is a way to express yourself. Gazpacho was a mafia figher \
    and driver of long distance trucks. He was a good driver. He was a good driver because he was a mafia fighter. \
    He was a mafia fighter because he was a good driver. He was a good driver because he was a mafia fighter. \
    "

    # Test the function
    paragraphs = autosplit_paragraphs(text, 50)

    assert len(paragraphs) == 5


if __name__ == "__main__":
    pytest.main()

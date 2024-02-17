"""
    GPT book is an gradio application which gives user :
    outout comparison is being shown.

    Testing capabilities of GPT3-4 models for text generation.
"""

import gradio as gr

from gpt_book.models.billings import Billings


def gradio_setup(
    ai_models: list, billings: Billings, text_process: callable
) -> gr.Interface:
    """
    Setup gradio application with
    - Gradio file input field
    - Gradio Combo box for OpenAI model selection,
    - Gradio Button to start processing of input file
    - Gradio HTML output where two <div's> side by side input and

    Using gradio blocks.
    """
    with gr.Blocks() as interface:
        # Title + Description
        gr.Markdown(
            "GPT Book - Create your own book from unstructured text!\n"
            + "-------------------\n"
            + "Testing capabilities of GPT models for book generation from free text without any constrains."
        )

        # Billings : Markdown
        gr.Markdown(f"Total cost of all API calls : {billings.total_cost} USD")

        # File Input : Input file
        # Gradio file input field
        input_file = gr.components.File(
            label="Please upload a text file for AI model to process as book."
        )

        # AI Models : Combo box
        # Gradio Combo box for OpenAI model selection
        ai_models_str = [
            (f"{model_name} / {token_size} tokens", model_name)
            for model_name, token_size in ai_models
        ]
        input_model = gr.components.Dropdown(
            label="Select OpenAI model ",
            choices=ai_models_str,
            value=ai_models_str[0],
        )

        # Button : Submit
        text_button = gr.Button("Submit")

        # Gradio HTML output where two <div's> side by side input and
        output_html = gr.components.HTML(label="Output")

        # Callback : When submit button is clicked
        text_button.click(
            lambda input_file, input_model: text_process(
                input_file, input_model, billings
            ),
            inputs=[
                input_file,
                input_model,
            ],
            outputs=[output_html],
        )

    return interface


def view_html(input_text: str, output_text: str) -> str:
    """Reads views/view.html file and replaces INPUT as input_text and OUTPUT as output_text"""
    with open("views/view.html", "r") as file:
        html = file.read()

    html = html.replace("INPUT", input_text).replace("OUTPUT", output_text)
    return html


def text_process(input_file: str, model: str, billings: Billings) -> str:
    """
    Process input file and return output comparison.
    """
    # Read input file
    with open(input_file.name, "r") as file:
        input_text = file.read()

    # Process input text using selected model
    output_text = f"Output from {model} model"

    # Return output comparison
    return view_html(input_text, output_text)


if __name__ == "__main__":
    # Billings : Create instance
    billings = Billings()
    # List of all AI models
    ai_models = [("gpt-3.5-turbo-0125", 16385), ("gpt-4", 8192)]

    # Gradio setup
    gradio_setup(
        ai_models=ai_models, billings=billings, text_process=text_process
    ).launch()

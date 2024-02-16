"""
    GPT book is an gradio application which gives user :
    outout comparison is being shown.

    Testing capabilities of GPT3-4 models for text generation.
"""

import gradio as gr


def gradio_setup(ai_models: list, text_process: callable) -> gr.Interface:
    """
    Setup gradio application with
    - Gradio file input field
    - Gradio Combo box for OpenAI model selection,
    - Gradio Button to start processing of input file
    - Gradio HTML output where two <div's> side by side input and

    Using gradio blocks.
    """

    # Gradio file input field
    input_file = gr.components.File(label="Input Text File")

    ai_models_str = [
        f"{model_name} / {token_size}" for model_name, token_size in ai_models
    ]

    # Gradio Combo box for OpenAI model selection
    model = gr.components.Dropdown(
        label="Model",
        choices=ai_models_str,
    )

    # Gradio HTML output where two <div's> side by side input and
    output = gr.components.HTML(label="Output")

    # Gradio Interface
    interface = gr.Interface(
        fn=text_process,
        inputs=[input_file, model],
        outputs=output,
        title="GPT Book",
        description="Testing capabilities of GPT3-4 models for text generation.",
    )

    return interface


def view_html(input_text: str, output_text: str) -> str:
    """Reads views/view.html file and replaces INPUT as input_text and OUTPUT as output_text"""
    with open("views/view.html", "r") as file:
        html = file.read()

    html = html.replace("INPUT", input_text).replace("OUTPUT", output_text)
    return html


def text_process(input_file: str, model: str) -> str:
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


def models_list():
    """Use Langchain openai to list all available chat models"""
    from langchain_openai import OpenAI

    openai = OpenAI()
    return openai.list_models()


if __name__ == "__main__":
    ai_models = [("gpt-3.5-turbo-0125", 16385), ("gpt-4", 8192)]
    gradio_setup(ai_models=ai_models, text_process=text_process).launch()

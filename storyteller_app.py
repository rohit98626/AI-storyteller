import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

print("Loading base model + LoRA adapter...")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "lora_storyteller")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("Model ready for Storytelling")

def generate_story(prompt):
    time.sleep(1) 
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_length=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    story = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return story

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="pink",
        neutral_hue="gray",
    ),
    css="""
    body {
        background: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    #title {
        text-align: center;
        font-size: 3em;
        background: linear-gradient(to right, #8e2de2, #4a00e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    #subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #ccc;
        margin-top: 0;
    }
    .gr-box {
        background: #1e1e1e;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.05);
    }
    .gr-button {
        background: linear-gradient(to right, #8e2de2, #4a00e0) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        padding: 12px 30px !important;
    }
    hr {
        margin-top: 40px;
        border: none;
        border-top: 2px solid #333;
    }
    """
) as storyteller_ui:

    gr.Markdown("<h1 id='title'>✨ Your Magical AI StoryTeller ✨</h1>")
    gr.Markdown("<p id='subtitle'>Give me your prompt, click the button, and let me craft a tale in the dark... 🌙</p>")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="📜 Your Story Prompt",
                placeholder="E.g., In a mystical forest under a blood moon there was a ...",
                lines=8,
                elem_classes="gr-box"
            )
            generate_btn = gr.Button("🌌 Tell Me a Story!", elem_classes="gr-button")
        with gr.Column(scale=1):
            story_output = gr.Textbox(
                label="📖 AI-Generated Story",
                placeholder="Your story will appear here...",
                lines=20,
                interactive=False,
                elem_classes="gr-box"
            )

    generate_btn.click(
        fn=generate_story,
        inputs=prompt_input,
        outputs=story_output,
        api_name="generate_story",
        show_progress=True
    )

    gr.Markdown("<hr>")

storyteller_ui.launch(server_name="127.0.0.1", server_port=7860)
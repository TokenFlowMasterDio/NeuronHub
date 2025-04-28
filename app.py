import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
MODEL_NAME = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Function to generate response
def generate_response(prompt, temperature, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio interface
def launch_interface():
    gr.Interface(
        fn=generate_response,
        inputs=[
            gr.Textbox(label="Enter your prompt"),
            gr.Slider(0.1, 1.0, step=0.01, value=0.7, label="Temperature"),
            gr.Slider(50, 512, step=10, value=200, label="Max New Tokens")
        ],
        outputs=gr.Textbox(label="Response"),
        title="🧠 NeuronHub",
        theme=gr.themes.Soft(),
        css=".footer {visibility: hidden;}"
    ).launch()

if __name__ == "__main__":
    launch_interface()

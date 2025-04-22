import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the TinyLlama model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Chat function
def chat(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 TinyLlama Chatbot (1.1B)")
    prompt = gr.Textbox(lines=3, label="Enter your prompt")
    temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
    max_tokens = gr.Slider(50, 512, value=200, step=10, label="Max New Tokens")
    output = gr.Textbox(lines=10, label="Response")
    submit = gr.Button("Generate")

    submit.click(chat, inputs=[prompt, max_tokens, temperature], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)

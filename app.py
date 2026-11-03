
# Name: Lathika M
# Register Number: 24BAD062
# Experiment No: 9 - Part C

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pre-trained model
model_name = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Chatbot function
def chatbot(message, history):

    # Prepare conversation history
    chat_text = ""

    for item in history[-6:]:
        if item["role"] == "user":
            chat_text += item["content"] + tokenizer.eos_token
        elif item["role"] == "assistant":
            chat_text += item["content"] + tokenizer.eos_token

    # Add current user message
    chat_text += message + tokenizer.eos_token

    # Tokenize input
    inputs = tokenizer.encode(
        chat_text,
        return_tensors="pt"
    )

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chatbot,
    type="messages",
    title="AI Chatbot - Lathika M",
    description="Register Number: 24BAD062 | Ask me anything!",
    chatbot=gr.Chatbot(height=400)
)

# Launch application
if __name__ == "__main__":
    demo.launch()

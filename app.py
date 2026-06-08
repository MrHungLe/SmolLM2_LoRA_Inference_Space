import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import gradio as gr

# 1. Model and Dataset configuration (configured in config.py)
from config import BASE_MODEL, LORA_ADAPTER, DATASET_NAME

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading Base Model (running on CPU)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("Loading LoRA Adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

# 2. Inference function
def generate_response(prompt, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the output response
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the response (you can strip the original prompt if you only want the generated response)
    return full_output

# 3. Fetch sample examples from your dataset to display on the UI
print("Loading dataset for suggestion examples...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    # Get the first 3 rows from the text column (or adjust the column name to match your dataset)
    # Assume the input data column is named 'text' or 'prompt'
    column_name = 'text' if 'text' in dataset.column_names else dataset.column_names[0]
    examples = [[dataset[i][column_name]] for i in range(min(3, len(dataset)))]
except Exception as e:
    print(f"Could not load dataset for examples: {e}. Using default examples instead.")
    examples = [["Write a short paragraph about AI technology."], ["Hello, who are you?"]]

# 4. Setup Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# CTU-ai-lab: SmolLM2 LoRA Inference Space")
    gr.Markdown(f"Application to test the fine-tuned model using Docker. Base: `{BASE_MODEL}` | Adapter: `{LORA_ADAPTER}`")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=4, label="Input / Prompt", placeholder="Type your prompt here...")
            max_tokens = gr.Slider(minimum=10, maximum=512, value=128, step=10, label="Max New Tokens")
            temp = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
            submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(lines=8, label="Model Response")
            
    submit_btn.click(
        fn=generate_response, 
        inputs=[input_text, max_tokens, temp], 
        outputs=output_text
    )
    
    # Add examples section from your dataset
    gr.Examples(examples=examples, inputs=input_text)

if __name__ == "__main__":
    # Hugging Face Spaces requires port 7860 and host 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=7860)
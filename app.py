import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import gradio as gr

# 1. Định nghĩa thông tin Model và Dataset của bạn (được cấu hình trong config.py)
from config import BASE_MODEL, LORA_ADAPTER, DATASET_NAME


print("Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Đang tải Base Model (chạy trên CPU)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("Đang nạp LoRA Adapter của bạn...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

# 2. Hàm xử lý suy luận (Inference)
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
    
    # Giải mã kết quả đầu ra
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Trả về kết quả (bạn có thể cắt bỏ phần prompt gốc nếu chỉ muốn hiển thị câu trả lời)
    return full_output

# 3. Lấy một vài ví dụ từ Dataset của bạn để hiển thị lên UI
print("Đang tải dataset để làm ví dụ gợi ý...")
try:
    dataset = load_dataset(DATASET_NAME, split="train")
    # Lấy 3 câu đầu tiên từ cột text (hoặc đổi tên cột phù hợp với cấu trúc dataset của bạn)
    # Giả định cột chứa dữ liệu đầu vào tên là 'text' hoặc 'prompt'
    column_name = 'text' if 'text' in dataset.column_names else dataset.column_names[0]
    examples = [[dataset[i][column_name]] for i in range(min(3, len(dataset)))]
except Exception as e:
    print(f"Không thể tải dataset làm ví dụ: {e}. Sẽ dùng ví dụ mặc định.")
    examples = [["Hãy viết một đoạn văn ngắn về công nghệ AI."], ["Hello, who are you?"]]

# 4. Thiết lập Giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# CTU-ai-lab: SmolLM2 LoRA Inference Space")
    gr.Markdown(f"Ứng dụng chạy thử nghiệm mô hình đã được tinh chỉnh bằng Docker. Base: `{BASE_MODEL}` | Adapter: `{LORA_ADAPTER}`")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=4, label="Nhập câu lệnh / Prompt", placeholder="Nhập văn bản tại đây...")
            max_tokens = gr.Slider(minimum=10, maximum=512, value=128, step=10, label="Max New Tokens")
            temp = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")
            submit_btn = gr.Button("Gửi yêu cầu", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(lines=8, label="Kết quả từ Mô hình")
            
    submit_btn.click(
        fn=generate_response, 
        inputs=[input_text, max_tokens, temp], 
        outputs=output_text
    )
    
    # Thêm phần ví dụ từ dataset của bạn
    gr.Examples(examples=examples, inputs=input_text)

if __name__ == "__main__":
    # Hugging Face Spaces yêu cầu chạy cố định ở port 7860 và host 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=7860)
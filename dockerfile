# Sử dụng Python base image phiên bản slim để giảm dung lượng dung lượng image
FROM python:3.9-slim

# Thiết lập thư mục làm việc mặc định trong container
WORKDIR /code

# Sao chép file requirements.txt vào trước để tận dụng Docker Cache
COPY ./requirements.txt /code/requirements.txt

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Tạo một User mới tên là 'user' để tuân thủ chính sách bảo mật của Hugging Face (không chạy quyền root)
RUN useradd -m -u 1000 user
USER user

# Thiết lập biến môi trường cho Home của User
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Sao chép toàn bộ mã nguồn ở máy bạn vào thư mục ứng dụng trong Container
COPY --chown=user . $HOME/app

# Tải trước Model và Tokenizer trong quá trình build để tăng tốc khởi động ứng dụng trên HF Spaces
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; from peft import PeftModel; from config import BASE_MODEL, LORA_ADAPTER; AutoTokenizer.from_pretrained(BASE_MODEL); base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL); PeftModel.from_pretrained(base_model, LORA_ADAPTER)"

# Mở cổng 7860 (Cổng bắt buộc của Hugging Face Spaces)
EXPOSE 7860

# Khởi chạy ứng dụng app.py khi Container khởi động
CMD ["python", "app.py"]
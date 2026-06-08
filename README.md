# SmolLM2-135M LoRA Inference App 🚀

Ứng dụng chạy thử nghiệm mô hình ngôn ngữ lớn **SmolLM2-135M** đã được tinh chỉnh bằng kỹ thuật **LoRA (Low-Rank Adaptation)**. Giao diện được xây dựng bằng **Gradio** và đóng gói bằng **Docker**.

## 📌 Các tính năng chính
- Xử lý suy luận (Inference) thời gian thực trên CPU với tùy chọn `max_new_tokens` và `temperature`.
- Tự động hiển thị các ví dụ mẫu lấy trực tiếp từ tập dữ liệu tinh chỉnh (`CTU-ai-lab/my-custom-dataset`).
- Đóng gói hoàn chỉnh bằng Docker giúp dễ dàng triển khai đa nền tảng (Local, Hugging Face Spaces, v.v.).
- Đã được tối ưu hóa tải trước Model (`pre-download`) trong quá trình build Docker giúp ứng dụng khởi động tức thì.

---

## 📂 Cấu trúc thư mục
```text
step9_intern_guideline/
├── config.py          # Cấu hình Model Base, LoRA Adapter và Dataset
├── app.py             # File chạy ứng dụng giao diện Gradio
├── dockerfile         # Cấu hình đóng gói Container
├── requirements.txt   # Các thư viện Python cần thiết
├── .gitignore         # Bỏ qua các file rác khi push lên Git
├── .dockerignore      # Bỏ qua các file không cần thiết khi build Docker
└── README.md          # Hướng dẫn này
```

---

## 🛠️ Hướng dẫn chạy cục bộ (Local)

### Cách 1: Chạy trực tiếp bằng Python
1. Tạo môi trường ảo và kích hoạt:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Trên macOS/Linux
   # hoặc: .venv\Scripts\activate  # Trên Windows
   ```
2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy ứng dụng:
   ```bash
   python app.py
   ```
4. Truy cập giao diện tại: `http://localhost:7860`

### Cách 2: Chạy với Docker
1. Build Docker image:
   ```bash
   docker build -t smollm2-app .
   ```
2. Chạy container:
   ```bash
   docker run -p 7860:7860 smollm2-app
   ```
3. Truy cập giao diện tại: `http://localhost:7860`

---

## 🌐 Triển khai (Deployment)

Dự án này đã sẵn sàng để triển khai lên **Hugging Face Spaces** hoặc **GitHub**. Vui lòng tham khảo tài liệu hướng dẫn chi tiết được cung cấp để push code lên các nền tảng này.

# SmolLM2-135M LoRA Inference App 🚀

An application to test and run inference for the **SmolLM2-135M** large language model fine-tuned using the **LoRA (Low-Rank Adaptation)** technique. The user interface is built using **Gradio** and packaged with **Docker**.

## 📌 Key Features
- Real-time inference on CPU with configurable parameters (`max_new_tokens` and `temperature`).
- Automatic fetching of suggestion examples directly from your fine-tuned dataset (`CTU-ai-lab/my-custom-dataset`).
- Fully containerized using Docker for seamless cross-platform deployment (Local, Hugging Face Spaces, etc.).
- Optimized startup speed by pre-downloading the model and adapter during the Docker build process, enabling instant launch on Hugging Face Spaces.

---

## 📂 Project Structure
```text
step9_intern_guideline/
├── config.py          # Model Base, LoRA Adapter, and Dataset configurations
├── app.py             # Gradio web interface and inference logic
├── dockerfile         # Docker container configuration (optimized with caching)
├── requirements.txt   # Required Python packages
├── .gitignore         # Prevents committing unnecessary local files to Git
├── .dockerignore      # Excludes local files from the Docker build context
└── README.md          # Project documentation (this file)
```

---

## 🛠️ Local Setup Instructions

### Option 1: Running directly with Python
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or: .venv\Scripts\activate  # On Windows
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   python app.py
   ```
4. Access the web interface at: `http://localhost:7860`

### Option 2: Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t smollm2-app .
   ```
2. Start the container:
   ```bash
   docker run -p 7860:7860 smollm2-app
   ```
3. Access the web interface at: `http://localhost:7860`

---

## 🌐 Deployment

This project is ready to be deployed to **Hugging Face Spaces** or pushed to **GitHub**. Refer to the deployment guide provided to push and share your results with your mentor.

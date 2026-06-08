# Use official Python slim image to reduce image size
FROM python:3.9-slim

# Set default working directory inside the container
WORKDIR /code

# Copy requirements.txt first to leverage Docker Cache
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a new user 'user' to comply with Hugging Face security guidelines (do not run as root)
RUN useradd -m -u 1000 user
USER user

# Set environment variables for User's home
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy application source code to container work directory
COPY --chown=user . $HOME/app

# Pre-download Model and Tokenizer during build time to accelerate application startup on HF Spaces
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; from peft import PeftModel; from config import BASE_MODEL, LORA_ADAPTER; AutoTokenizer.from_pretrained(BASE_MODEL); base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL); PeftModel.from_pretrained(base_model, LORA_ADAPTER)"

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Launch the application when container starts
CMD ["python", "app.py"]
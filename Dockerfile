# 1. Use a lightweight Python version
FROM python:3.11-slim

# 2. Install Linux system dependencies required by 'unstructured' and 'pdfplumber'
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 3. Hugging Face requires a non-root user named "user" with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 4. Set the working directory
WORKDIR $HOME/app

# 5. Copy and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy your backend code
COPY --chown=user app/ ./app/

# 7. Expose the Hugging Face port
EXPOSE 7860

# 8. Start the FastAPI backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
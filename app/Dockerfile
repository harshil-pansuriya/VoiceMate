# Use official Python 3.11 slim image
FROM python:3.11.4

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libpq-dev \
    gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create logs directory and set permissions
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Create cache directories and set permissions
RUN mkdir -p /app/.cache /tmp/.cache/hub && chmod -R 777 /app/.cache /tmp/.cache

# Set cache directory environment variables
ENV XDG_CACHE_HOME=/app/.cache
ENV HF_HOME=/app/.cache/hub
ENV TRANSFORMERS_CACHE=/app/.cache/hub

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Expose port 7860 (required by Hugging Face)
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
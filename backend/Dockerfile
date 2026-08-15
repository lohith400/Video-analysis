# IRIS Backend — FastAPI + YOLOv8 + EasyOCR
# Build (CPU, portable/demo):   docker build -t iris-backend .
# Build (GPU, needs nvidia-container-toolkit on host + CUDA base image):
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 -t iris-backend:gpu .

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# System deps required by opencv-python and easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Model weights are NOT baked into the image — mount them at runtime.
# See docker-compose.yml: ./backend/models -> /app/models
RUN mkdir -p /app/models /app/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

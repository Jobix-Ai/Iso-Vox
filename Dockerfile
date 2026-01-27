# Multi-stage build for Vietnamese-English ML STT service
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    ca-certificates \
    gnutls-bin \
    libgnutls30 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    sox \
    && update-ca-certificates \
    && python3.11 --version \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Configure git for better SSL/TLS handling
RUN git config --global http.postBuffer 524288000 \
    && git config --global http.sslVerify true

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip to latest version
RUN pip3 install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/moonshine-onnx ./src/moonshine-onnx

# Copy pre-cloned moonshine repository and install
RUN cd src/moonshine-onnx && pip3 install . && cd ../..
RUN pip uninstall -y onnxruntime
RUN pip install -U onnxruntime-gpu

# Copy application files
COPY src/*.py ./src/
COPY *.sh ./

# Copy ONNX model (required for smart_turn_worker_app)
COPY weights ./weights
#COPY DNSMOSPro ./DNSMOSPro

# Copy health check script
COPY .health-check.sh /health-check.sh

# Make shell scripts executable
RUN chmod +x *.sh /health-check.sh

# Create necessary directories
RUN mkdir -p logs debug_audio

# Expose ports
# 8089 - Gateway
# 8090 - ASR Batching Worker  
# 8091 - Smart Turn Worker
# 8092 - Speaker Verification
# 8093 - Parakeet ASR Worker
# 8094 - Moonshine ASR Worker
EXPOSE 8089 8090 8091 8092 8093 8094

# Create entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Override NVIDIA entrypoint and set our custom entrypoint
# The NVIDIA entrypoint will be bypassed, and our entrypoint will handle everything
ENTRYPOINT ["/entrypoint.sh"]
CMD ["all"]


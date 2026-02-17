FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11, Node.js, ffmpeg, and espeak-ng (required by Kokoro for phonemes)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    ffmpeg \
    libavcodec-extra \
    espeak-ng \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev libsndfile1 gcc \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch + CUDA from pinned wheels (never re-resolves, always caches)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl \
    https://download.pytorch.org/whl/cu126/torchaudio-2.6.0%2Bcu126-cp311-cp311-linux_x86_64.whl

# Install remaining Python dependencies (invalidated when requirements.txt changes)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy and install Node.js dependencies
COPY audio-server/package*.json ./audio-server/
WORKDIR /app/audio-server
RUN npm install

# Copy the rest of the application
WORKDIR /app
COPY . .

# Create uploads and outputs directories
RUN mkdir -p /app/audio-server/uploads /app/audio-server/outputs

# Expose ports (3000 = HTTP/WS, 9001 = internal STT WebSocket)
EXPOSE 3000 9001

# Start the server
WORKDIR /app/audio-server
CMD ["node", "server.js"]

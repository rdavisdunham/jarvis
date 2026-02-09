FROM python:3.11-slim

# Install Node.js and ffmpeg with full codec support
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libavcodec-extra \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
# 1. Copy the 'uv' binary directly from the official image
# (This is faster and cleaner than 'pip install uv')
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 2. Install dependencies using uv + BuildKit Cache
# The --mount flag creates the "wormhole" to your host machine at /root/.cache/uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy and install Node.js dependencies
COPY audio-server/package*.json ./audio-server/
WORKDIR /app/audio-server
RUN npm install

# Copy the rest of the application
WORKDIR /app
COPY . .

# Create uploads and outputs directories
RUN mkdir -p /app/audio-server/uploads /app/audio-server/outputs

# Expose port
EXPOSE 3000

# Start the server
WORKDIR /app/audio-server
CMD ["node", "server.js"]

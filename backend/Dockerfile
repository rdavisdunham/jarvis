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
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install Node.js dependencies
COPY audio-server/package*.json ./audio-server/
WORKDIR /app/audio-server
RUN npm install

# Copy the rest of the application
WORKDIR /app
COPY . .

# Create uploads directory
RUN mkdir -p /app/audio-server/uploads

# Expose port
EXPOSE 3000

# Start the server
WORKDIR /app/audio-server
CMD ["node", "server.js"]

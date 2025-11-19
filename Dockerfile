# Dockerfile for robust_tiff_compress webserver
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for imagecodecs and tifffile
RUN apt-get update && apt-get install -y \
    build-essential \
    libz-dev \
    libjpeg-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY robust_tiff_compress.py .
COPY webserver/ ./webserver/
COPY main.py .

# Create /data directory for mounting
RUN mkdir -p /data

# Expose webserver port
EXPOSE 8000

# Set environment variable for port (can be overridden)
ENV PORT=8000
ENV PRESERVE_OWNERSHIP=1

# Run webserver
# Use sh to read PORT environment variable
CMD sh -c "uvicorn webserver.server:app --host 0.0.0.0 --port ${PORT:-8000}"


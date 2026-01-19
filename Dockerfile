# Dockerfile for Resume NER API
# This Dockerfile builds a containerized API server for the Resume NER model.
#
# Requires Docker BuildKit (enabled by default in Docker 18.09+)
# To enable: export DOCKER_BUILDKIT=1
# Or use: DOCKER_BUILDKIT=1 docker build ...

FROM python:3.10-slim

# Build arguments for model paths
# These can be overridden when building the image:
#   docker build --build-arg ONNX_MODEL_PATH=path/to/model.onnx --build-arg CHECKPOINT_PATH=path/to/checkpoint .
# If not provided, model files must be mounted as volumes at runtime
ARG ONNX_MODEL_PATH
ARG CHECKPOINT_PATH

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create models directory
RUN mkdir -p /app/models/checkpoint

# Copy model files
# 
# IMPORTANT: Model files can be provided in two ways:
#   1. Build arguments (baked into image): 
#      docker build --build-arg ONNX_MODEL_PATH=path/to/model.onnx --build-arg CHECKPOINT_PATH=path/to/checkpoint .
#   2. Volume mounts at runtime (recommended for development):
#      docker run -v /path/to/model.onnx:/app/models/model.onnx -v /path/to/checkpoint:/app/models/checkpoint ...
#
# If build arguments are not provided, the COPY commands below will be skipped.
# You must then mount the model files as volumes when running the container.

# Copy ONNX model (only if ONNX_MODEL_PATH build arg is provided)
# The model file will be available at /app/models/model.onnx in the container
RUN --mount=type=bind,source=.,target=/buildcontext \
    if [ -n "$ONNX_MODEL_PATH" ] && [ -f "/buildcontext/$ONNX_MODEL_PATH" ]; then \
      cp "/buildcontext/$ONNX_MODEL_PATH" /app/models/model.onnx && \
      echo "Copied ONNX model from $ONNX_MODEL_PATH"; \
    else \
      echo "ONNX_MODEL_PATH not provided or file not found - mount model.onnx as volume at runtime"; \
      touch /app/models/model.onnx; \
    fi

# Copy checkpoint directory (only if CHECKPOINT_PATH build arg is provided)
# IMPORTANT: The checkpoint directory must contain:
#   - config.json (with id2label mapping)
#   - tokenizer.json or tokenizer_config.json  
#   - vocab.txt or similar vocabulary files
RUN --mount=type=bind,source=.,target=/buildcontext \
    if [ -n "$CHECKPOINT_PATH" ] && [ -d "/buildcontext/$CHECKPOINT_PATH" ]; then \
      cp -r "/buildcontext/$CHECKPOINT_PATH"/* /app/models/checkpoint/ && \
      echo "Copied checkpoint from $CHECKPOINT_PATH"; \
    else \
      echo "CHECKPOINT_PATH not provided or directory not found - mount checkpoint as volume at runtime"; \
    fi

# Set environment variables
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_WORKERS=1
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check (using curl which is more reliable in containers)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "-m", "src.deployment.api.cli.run_api", \
     "--onnx-model", "/app/models/model.onnx", \
     "--checkpoint", "/app/models/checkpoint", \
     "--host", "0.0.0.0", \
     "--port", "8000"]


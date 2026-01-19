# Docker Installation Guide

This guide explains how to build and run the Resume NER API using Docker.

## Overview

The Resume NER API is containerized using Docker and requires:
- An ONNX model file (`model.onnx`)
- A checkpoint directory containing tokenizer and configuration files

The Dockerfile supports flexible model paths through:
- **Build arguments**: Model files are baked into the Docker image (recommended for production)
- **Volume mounts**: Model files are mounted at runtime (recommended for development)

## Quick Start

**With build arguments (model files in image):**
```bash
docker build \
  --build-arg ONNX_MODEL_PATH=path/to/model.onnx \
  --build-arg CHECKPOINT_PATH=path/to/checkpoint \
  -t resume-ner-api:latest .

docker run -d -p 8000:8000 resume-ner-api:latest
```

**With volume mounts (model files from host):**
```bash
docker build -t resume-ner-api:latest .

docker run -d -p 8000:8000 \
  -v $(pwd)/path/to/model.onnx:/app/models/model.onnx \
  -v $(pwd)/path/to/checkpoint:/app/models/checkpoint \
  resume-ner-api:latest
```

## Prerequisites

- Docker installed and running
- Docker Compose (optional, for easier management)
- Access to the model files:
  - ONNX model file (`model.onnx`)
  - Checkpoint directory with tokenizer and config files

## Model Files Structure

The Docker container expects the following structure:

```
/app/models/
├── model.onnx          # ONNX model file
└── checkpoint/         # Checkpoint directory with tokenizer files
    ├── config.json     # Model configuration (must contain id2label mapping)
    ├── tokenizer.json  # Tokenizer configuration (or tokenizer_config.json)
    ├── tokenizer_config.json
    └── vocab.txt       # Vocabulary file (if applicable)
```

### Finding the Checkpoint Directory

The checkpoint directory is typically located in your training outputs. It should be the same directory that was used during model training and contains the tokenizer files saved by HuggingFace Transformers.

Common locations:
- Training output directory (e.g., `outputs/training/.../checkpoint-*/`)
- The original model checkpoint before conversion
- A directory containing `config.json` and tokenizer files

**Important**: The checkpoint directory must contain:
- `config.json` with the `id2label` mapping (required for entity decoding)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)
- Vocabulary files (`vocab.txt` or similar, depending on the tokenizer type)

## Building the Docker Image

### Step 1: Prepare Model Files

Before building, ensure you have:
1. An ONNX model file (`model.onnx`) - the converted model for inference
2. A checkpoint directory with tokenizer files (typically from the training output)

**Note**: The checkpoint directory should contain the tokenizer files needed by `transformers.AutoTokenizer.from_pretrained()`. This is usually the same directory used during model training.

### Step 2: Build Options

**Note**: This Dockerfile uses Docker BuildKit features. BuildKit is enabled by default in Docker 18.09+. If you're using an older version, enable it with:
```bash
export DOCKER_BUILDKIT=1
```

You have three options for providing model files:

#### Option A: Build Arguments (Recommended for Production)

Build the image with model paths as build arguments. The model files will be baked into the image:

```bash
docker build \
  --build-arg ONNX_MODEL_PATH=path/to/your/model.onnx \
  --build-arg CHECKPOINT_PATH=path/to/your/checkpoint \
  -t resume-ner-api:latest .
```

**Note**: Paths are relative to the build context (the directory where you run `docker build`).

**Example with specific paths:**
```bash
docker build \
  --build-arg ONNX_MODEL_PATH=outputs/conversion/local/distilbert/spec-1e6acb58_exec-2cfc5e4f/v1/conv-6781b0fa/onnx_model/model.onnx \
  --build-arg CHECKPOINT_PATH=outputs/training/checkpoint-1000 \
  -t resume-ner-api:latest .
```

#### Option B: Volume Mounts (Recommended for Development)

Build without model files and mount them at runtime:

```bash
# Build the image
docker build -t resume-ner-api:latest .

# Run with volume mounts (see "Run with Volume Mounts" section below)
```

#### Option C: Modify Dockerfile

Edit the Dockerfile directly to hardcode your model paths:

```dockerfile
COPY path/to/your/model.onnx /app/models/model.onnx
COPY path/to/your/checkpoint /app/models/checkpoint/
```

Then build normally:

```bash
docker build -t resume-ner-api:latest .
```

### Step 3: Build the Image

Choose one of the options above and build:

```bash
# With build arguments
docker build \
  --build-arg ONNX_MODEL_PATH=path/to/model.onnx \
  --build-arg CHECKPOINT_PATH=path/to/checkpoint \
  -t resume-ner-api:latest .

# Or without build arguments (for volume mounting)
docker build -t resume-ner-api:latest .
```

## Running the Container

### Basic Run

```bash
docker run -d \
  --name resume-ner-api \
  -p 8000:8000 \
  resume-ner-api:latest
```

### Run with Custom Configuration

```bash
docker run -d \
  --name resume-ner-api \
  -p 8000:8000 \
  -e API_PORT=8080 \
  -e API_WORKERS=2 \
  -e LOG_LEVEL=DEBUG \
  resume-ner-api:latest
```

### Run with Volume Mounts (for development)

If you want to mount model files from your host (useful when model files weren't included in the image):

```bash
docker run -d \
  --name resume-ner-api \
  -p 8000:8000 \
  -v /absolute/path/to/model.onnx:/app/models/model.onnx \
  -v /absolute/path/to/checkpoint:/app/models/checkpoint \
  resume-ner-api:latest
```

**Example:**
```bash
docker run -d \
  --name resume-ner-api \
  -p 8000:8000 \
  -v $(pwd)/outputs/conversion/local/distilbert/spec-1e6acb58_exec-2cfc5e4f/v1/conv-6781b0fa/onnx_model/model.onnx:/app/models/model.onnx \
  -v $(pwd)/outputs/training/checkpoint-1000:/app/models/checkpoint \
  resume-ner-api:latest
```

### Run with GPU Support (if available)

For GPU acceleration, use NVIDIA Docker runtime:

```bash
docker run -d \
  --name resume-ner-api \
  --gpus all \
  -p 8000:8000 \
  -e ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider" \
  resume-ner-api:latest
```

**Note**: For GPU support, you may need to install `onnxruntime-gpu` instead of `onnxruntime` in the requirements.txt.

## Environment Variables

The following environment variables can be configured:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Host to bind the API server |
| `API_PORT` | `8000` | Port to bind the API server |
| `API_WORKERS` | `1` | Number of worker processes |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `MAX_FILE_SIZE` | `10485760` | Maximum file upload size in bytes (10MB) |
| `MAX_BATCH_SIZE` | `32` | Maximum batch size for batch predictions |
| `MAX_SEQUENCE_LENGTH` | `512` | Maximum sequence length for tokenization |
| `ONNX_PROVIDERS` | `CPUExecutionProvider` | Comma-separated list of ONNX Runtime providers |
| `CORS_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `PDF_EXTRACTOR` | `pymupdf` | PDF extractor to use (`pymupdf` or `pdfplumber`) |
| `OCR_EXTRACTOR` | `easyocr` | OCR extractor to use (`easyocr` or `pytesseract`) |

## Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  resume-ner-api:
    build: .
    container_name: resume-ner-api
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - API_WORKERS=1
      - LOG_LEVEL=INFO
    volumes:
      # Optional: mount model files from host
      # - ./outputs/conversion/local/distilbert/spec-1e6acb58_exec-2cfc5e4f/v1/conv-6781b0fa/onnx_model/model.onnx:/app/models/model.onnx
      # - ./path/to/checkpoint:/app/models/checkpoint
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Then run:

```bash
docker-compose up -d
```

## Verifying the Installation

### Check Container Status

```bash
docker ps | grep resume-ner-api
```

### Check Logs

```bash
docker logs resume-ner-api
```

### Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Test Model Info Endpoint

```bash
curl http://localhost:8000/info
```

### Test Prediction Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "John Doe is a Software Engineer at Microsoft in Seattle."}'
```

## Troubleshooting

### Container Fails to Start

1. **Check logs**:
   ```bash
   docker logs resume-ner-api
   ```

2. **Verify model files exist**:
   ```bash
   docker exec resume-ner-api ls -la /app/models/
   ```

3. **Check if checkpoint directory has required files**:
   ```bash
   docker exec resume-ner-api ls -la /app/models/checkpoint/
   ```

### Model Not Loading

- Ensure the checkpoint directory contains:
  - `config.json` (with `id2label` mapping)
  - `tokenizer.json` or `tokenizer_config.json`
  - Vocabulary files (`vocab.txt` or similar)

- Check the API logs for specific error messages:
  ```bash
  docker logs resume-ner-api | grep -i error
  ```

### Port Already in Use

If port 8000 is already in use, change the port:

```bash
docker run -d \
  --name resume-ner-api \
  -p 8080:8000 \
  resume-ner-api:latest
```

Then access the API at `http://localhost:8080`

### Memory Issues

If you encounter memory issues, try:
- Reducing `API_WORKERS` to 1
- Reducing `MAX_BATCH_SIZE`
- Using CPU-only execution (default)

## Production Deployment

For production deployment, consider:

1. **Use a reverse proxy** (nginx, Traefik) in front of the container
2. **Set up proper logging** (mount logs directory or use logging service)
3. **Configure resource limits**:
   ```bash
   docker run -d \
     --name resume-ner-api \
     --memory="2g" \
     --cpus="2" \
     -p 8000:8000 \
     resume-ner-api:latest
   ```
4. **Use environment-specific configurations**
5. **Set up monitoring and alerting**
6. **Use container orchestration** (Kubernetes, Docker Swarm) for scaling

## Stopping and Removing

```bash
# Stop the container
docker stop resume-ner-api

# Remove the container
docker rm resume-ner-api

# Remove the image (optional)
docker rmi resume-ner-api:latest
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Docker Documentation](https://docs.docker.com/)


# Deployment Guide

## Overview

This guide covers deploying the Resume NER API service in various environments, from local development to production.

## Prerequisites

- Python 3.8+
- ONNX model file (`distilroberta_model.onnx` or similar)
- Checkpoint directory with tokenizer and model config
- Required Python packages (see `requirements.txt`)

## Local Development Setup

### 1. Install Dependencies

```bash
# Install core dependencies
pip install fastapi uvicorn[standard] pydantic onnxruntime transformers

# Install text extraction dependencies (optional, based on needs)
pip install pymupdf  # For PDF extraction
pip install easyocr  # For OCR (or pytesseract)
pip install pillow  # For image processing
```

Or install from requirements file:

```bash
pip install -r requirements-api.txt  # If you create this file
```

### 2. Prepare Model Files

Ensure you have:
- ONNX model file (e.g., `outputs/conversion/distilroberta/model_int8.onnx`)
- Checkpoint directory (e.g., `outputs/final_training/distilroberta/checkpoint`)

### 3. Start the Server

```bash
```bash
python -m src.api.cli.run_api \
  --onnx-model outputs/conversion/distilroberta/model_int8.onnx \
  --checkpoint outputs/final_training/distilroberta/checkpoint \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level INFO
```
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "John Doe is a software engineer."}'
```

## Environment Variables

Configure the API using environment variables:

```bash
# Model paths
export ONNX_MODEL_PATH="outputs/conversion/distilroberta/model_int8.onnx"
export CHECKPOINT_DIR="outputs/final_training/distilroberta/checkpoint"

# Server settings
export API_HOST="0.0.0.0"
export API_PORT="8000"
export API_WORKERS="1"

# File upload limits
export MAX_FILE_SIZE="10485760"  # 10MB in bytes
export MAX_BATCH_SIZE="32"

# Text extraction
export PDF_EXTRACTOR="pymupdf"  # or "pdfplumber"
export OCR_EXTRACTOR="easyocr"  # or "pytesseract"

# Logging
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# CORS
export CORS_ORIGINS="*"  # or comma-separated list
export CORS_ALLOW_CREDENTIALS="false"
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "-m", "src.api.cli.run_api", \
     "--onnx-model", "/app/models/model.onnx", \
     "--checkpoint", "/app/models/checkpoint", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

### 2. Build Docker Image

```bash
docker build -t resume-ner-api:latest .
```

### 3. Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v /path/to/models:/app/models \
  --name resume-ner-api \
  resume-ner-api:latest
```

### 4. Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - ONNX_MODEL_PATH=/app/models/model.onnx
      - CHECKPOINT_DIR=/app/models/checkpoint
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

## Production Deployment

### Considerations

1. **Model Loading**: Models are loaded at startup. For large models, consider:
   - Using model quantization (INT8)
   - Lazy loading (load on first request)
   - Model caching

2. **Performance**:
   - Use multiple workers (`--workers N`)
   - Enable GPU support (use `onnxruntime-gpu`)
   - Consider using a reverse proxy (nginx) for load balancing

3. **Security**:
   - Add authentication (API keys, OAuth2)
   - Enable HTTPS/TLS
   - Implement rate limiting
   - Validate and sanitize inputs

4. **Monitoring**:
   - Add logging to external service (e.g., ELK stack)
   - Set up health check monitoring
   - Track API metrics (latency, throughput, errors)

5. **Scalability**:
   - Use container orchestration (Kubernetes, Docker Swarm)
   - Implement horizontal scaling
   - Use message queues for async processing

### Example: Production Setup with Nginx

**Nginx Configuration** (`/etc/nginx/sites-available/resume-ner-api`):

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # File upload size
        client_max_body_size 10M;
    }
}
```

### Example: Systemd Service

Create `/etc/systemd/system/resume-ner-api.service`:

```ini
[Unit]
Description=Resume NER API Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/resume-ner-api
Environment="ONNX_MODEL_PATH=/opt/models/model.onnx"
Environment="CHECKPOINT_DIR=/opt/models/checkpoint"
ExecStart=/usr/bin/python3 -m src.api.cli.run_api \
    --onnx-model /opt/models/model.onnx \
    --checkpoint /opt/models/checkpoint \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable resume-ner-api
sudo systemctl start resume-ner-api
sudo systemctl status resume-ner-api
```

## Cloud Deployment

### AWS (EC2/ECS)

1. **EC2 Deployment**:
   - Launch EC2 instance
   - Install dependencies
   - Use systemd service (see above)
   - Configure security groups

2. **ECS Deployment**:
   - Create ECR repository
   - Push Docker image
   - Create ECS task definition
   - Deploy to ECS cluster

### Azure (Container Instances/App Service)

1. **Container Instances**:
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name resume-ner-api \
     --image resume-ner-api:latest \
     --ports 8000 \
     --environment-variables \
       ONNX_MODEL_PATH=/app/models/model.onnx \
       CHECKPOINT_DIR=/app/models/checkpoint
   ```

2. **App Service**:
   - Create App Service plan
   - Deploy container
   - Configure environment variables

### Google Cloud (Cloud Run)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/resume-ner-api

# Deploy to Cloud Run
gcloud run deploy resume-ner-api \
  --image gcr.io/PROJECT_ID/resume-ner-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ONNX_MODEL_PATH=/app/models/model.onnx
```

## Monitoring and Logging

### Health Checks

Set up monitoring to check `/health` endpoint:

```bash
# Simple health check script
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ $response -eq 200 ]; then
    echo "API is healthy"
    exit 0
else
    echo "API is unhealthy"
    exit 1
fi
```

### Logging

The API logs requests and responses. Configure logging:

```python
# In your deployment, configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

## Troubleshooting

### Model Not Loading

- Check file paths are correct
- Verify ONNX model file exists and is valid
- Check checkpoint directory contains tokenizer files
- Review logs for specific error messages

### High Memory Usage

- Use quantized models (INT8)
- Reduce batch size
- Limit concurrent requests
- Consider model optimization

### Slow Inference

- Enable GPU support (`onnxruntime-gpu`)
- Use quantized models
- Optimize sequence length
- Consider model caching

### File Upload Errors

- Check file size limits
- Verify file type is supported
- Ensure extractor dependencies are installed
- Check disk space

## Performance Tuning

1. **ONNX Runtime Providers**:
   - CPU: `CPUExecutionProvider`
   - GPU: `CUDAExecutionProvider` (requires `onnxruntime-gpu`)

2. **Worker Processes**:
   - Single worker: Good for development
   - Multiple workers: Better for production (match CPU cores)

3. **Batch Processing**:
   - Process multiple requests in parallel
   - Use async/await for I/O-bound operations

## Security Best Practices

1. **Authentication**: Add API key or OAuth2
2. **HTTPS**: Use TLS/SSL certificates
3. **Input Validation**: Validate all inputs
4. **Rate Limiting**: Prevent abuse
5. **CORS**: Restrict allowed origins
6. **File Upload**: Validate file types and sizes
7. **Error Messages**: Don't expose sensitive information

## Next Steps

- Set up CI/CD pipeline
- Configure monitoring and alerting
- Implement authentication
- Add rate limiting
- Set up backup and recovery procedures



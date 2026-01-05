# FastAPI Server Running Guide

## Overview

This guide explains how to start and run the Resume NER FastAPI inference server locally. The server provides REST API endpoints for extracting named entities from resume text, PDFs, and images using ONNX models.

## Prerequisites

### Required Dependencies

Install the following Python packages:

```bash
pip install fastapi uvicorn[standard] onnxruntime transformers python-multipart pyyaml
```

**Optional dependencies** (for file processing):

- **PDF extraction**: `pymupdf` or `pdfplumber`
- **Image OCR**: `easyocr` or `pytesseract` + `pillow`

```bash
# For PDF processing
pip install pymupdf  # or pdfplumber

# For image OCR
pip install easyocr  # or pytesseract pillow
```

### Required Files

Before starting the server, ensure you have:

1. **ONNX Model File**: Trained model in ONNX format (e.g., `distilroberta_model.onnx`)
2. **Checkpoint Directory**: Directory containing tokenizer and model configuration files

Typical location:

```
outputs/final_training/distilroberta/
├── distilroberta_model.onnx
└── checkpoint/
    ├── tokenizer.json
    ├── config.json
    └── ...
```

## Basic Server Startup

### Command Syntax

```bash
python -m src.api.cli.run_api \
    --onnx-model <path_to_onnx_model> \
    --checkpoint <path_to_checkpoint_dir>
```

### Example

```bash
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint
```

### Default Configuration

If not specified, the server uses these defaults:

- **Host**: `127.0.0.1` (localhost)
- **Port**: `8000`
- **Workers**: `1` (single worker)
- **Log Level**: `INFO`

The server will be accessible at: `http://127.0.0.1:8000`

## Command-Line Options

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--onnx-model` | Path to ONNX model file | `--onnx-model model.onnx` |
| `--checkpoint` | Path to checkpoint directory | `--checkpoint ./checkpoint` |

### Optional Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--host` | Server host address | `127.0.0.1` | `--host 0.0.0.0` |
| `--port` | Server port number | `8000` | `--port 8080` |
| `--workers` | Number of worker processes | `1` | `--workers 4` |
| `--log-level` | Logging level | `INFO` | `--log-level DEBUG` |
| `--reload` | Enable auto-reload (development) | `False` | `--reload` |

### Log Levels

Available log levels:

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages only

## Common Usage Examples

### 1. Basic Server (Default Settings)

```bash
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint
```

Server starts at: `http://127.0.0.1:8000`

### 2. Custom Host and Port

```bash
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    --host 0.0.0.0 \
    --port 8080
```

Server starts at: `http://0.0.0.0:8080` (accessible from other machines)

### 3. Development Mode (Auto-Reload)

```bash
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    --reload \
    --log-level DEBUG
```

**Note**: `--reload` automatically restarts the server when code changes are detected. Only works with a single worker.

### 4. Production Mode (Multiple Workers)

```bash
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    --workers 4 \
    --log-level INFO
```

**Note**: Multiple workers improve throughput but increase memory usage. Do not use `--reload` with multiple workers.

### 5. Debug Mode

```bash
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    --log-level DEBUG
```

## Verifying Server Status

### Health Check

Once the server is running, verify it's working:

```bash
# Using curl
curl http://127.0.0.1:8000/health

# Expected response:
# {"status":"ok","model_loaded":true,"message":"Service is running"}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: <http://127.0.0.1:8000/docs>
- **ReDoc**: <http://127.0.0.1:8000/redoc>

Open these URLs in your browser to:

- View all available endpoints
- Test API calls directly
- See request/response schemas

### Model Information

Check model details:

```bash
curl http://127.0.0.1:8000/info
```

## Server Output

### Successful Startup

When the server starts successfully, you'll see output like:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Model Loading

The server loads the model during startup. You'll see:

```
INFO:     Loading ONNX model from: outputs/final_training/distilroberta/distilroberta_model.onnx
INFO:     Model loaded successfully
INFO:     Tokenizer loaded from: outputs/final_training/distilroberta/checkpoint
```

## Stopping the Server

To stop the server:

1. **Press `CTRL+C`** in the terminal where the server is running
2. The server will perform a graceful shutdown:

```
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [12345]
```

## Troubleshooting

### Port Already in Use

**Error**: `Address already in use` or `Port 8000 is already in use`

**Solutions**:

1. **Use a different port**:

   ```bash
   python -m src.api.cli.run_api \
       --onnx-model <path> \
       --checkpoint <path> \
       --port 8001
   ```

2. **Find and kill the process using the port** (Windows):

   ```powershell
   # Find process
   netstat -ano | findstr :8000
   
   # Kill process (replace PID with actual process ID)
   taskkill /F /PID <PID>
   ```

3. **Find and kill the process using the port** (Linux/Mac):

   ```bash
   # Find process
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>
   ```

### Model File Not Found

**Error**: `Error: ONNX model not found: <path>`

**Solutions**:

1. **Verify the path is correct**:

   ```bash
   # Check if file exists
   ls -la outputs/final_training/distilroberta/distilroberta_model.onnx
   ```

2. **Use absolute path**:

   ```bash
   python -m src.api.cli.run_api \
       --onnx-model /absolute/path/to/model.onnx \
       --checkpoint /absolute/path/to/checkpoint
   ```

3. **Check current working directory**:

   ```bash
   # Make sure you're in the repository root
   pwd
   # Should be: /path/to/resume-ner-azureml
   ```

### Checkpoint Directory Not Found

**Error**: `Error: Checkpoint directory not found: <path>`

**Solutions**:

1. **Verify the checkpoint directory exists**:

   ```bash
   ls -la outputs/final_training/distilroberta/checkpoint
   ```

2. **Check required files are present**:

   ```bash
   # Should contain at least:
   # - tokenizer.json
   # - config.json
   ls outputs/final_training/distilroberta/checkpoint/
   ```

### Model Loading Fails

**Error**: Model fails to load or server crashes during startup

**Solutions**:

1. **Check ONNX model file integrity**:

   ```python
   import onnxruntime as ort
   session = ort.InferenceSession("path/to/model.onnx")
   ```

2. **Verify dependencies are installed**:

   ```bash
   pip list | grep onnxruntime
   # Should show: onnxruntime
   ```

3. **Check log level for detailed errors**:

   ```bash
   python -m src.api.cli.run_api \
       --onnx-model <path> \
       --checkpoint <path> \
       --log-level DEBUG
   ```

### Server Not Responding

**Symptoms**: Server starts but health check fails

**Solutions**:

1. **Check server logs** for errors
2. **Verify model loaded successfully**:

   ```bash
   curl http://127.0.0.1:8000/health
   # Should return: {"status":"ok","model_loaded":true}
   ```

3. **Check firewall settings** (if using `0.0.0.0` host)
4. **Try accessing from localhost only** (`127.0.0.1`)

### Memory Issues

**Symptoms**: Server crashes or becomes unresponsive under load

**Solutions**:

1. **Reduce number of workers**:

   ```bash
   python -m src.api.cli.run_api \
       --onnx-model <path> \
       --checkpoint <path> \
       --workers 1
   ```

2. **Monitor memory usage**:

   ```bash
   # Windows
   tasklist | findstr python
   
   # Linux/Mac
   ps aux | grep python
   ```

3. **Process requests in smaller batches**

## Performance Considerations

### Single Worker vs Multiple Workers

- **Single Worker (`--workers 1`)**:
  - Lower memory usage
  - Suitable for development
  - Better for debugging

- **Multiple Workers (`--workers 4`)**:
  - Higher throughput
  - Better for production
  - Higher memory usage (each worker loads the model)

### Model Loading Time

The first request may be slower due to:

- Model initialization
- ONNX runtime warmup
- Tokenizer loading

Subsequent requests will be faster.

### File Processing

PDF and image processing requires additional dependencies:

- **PDFs**: Install `pymupdf` or `pdfplumber`
- **Images**: Install `easyocr` or `pytesseract` + `pillow`

Without these, file upload endpoints will return `400 Bad Request`.

## Integration with Tests

The server can be started programmatically for testing. See:

- `tests/integration/api/test_helpers.py` - Server management utilities
- `tests/integration/api/test_api_local_server.py` - Integration tests

Example test usage:

```python
from tests.integration.api.test_helpers import ServerManager

server_manager = ServerManager()
handle = server_manager.start_server(
    onnx_path=Path("model.onnx"),
    checkpoint_dir=Path("checkpoint"),
    host="127.0.0.1",
    port=8000
)

# Use server...
# ...

server_manager.stop_server(handle)
```

## Next Steps

- **API Usage**: See [API_USAGE.md](API_USAGE.md) for endpoint documentation
- **Testing**: See [tests/README.md](../tests/README.md) for testing guide
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment

## Quick Reference

```bash
# Start server (basic)
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint

# Start server (custom port)
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    --port 8080

# Start server (development mode)
python -m src.api.cli.run_api \
    --onnx-model outputs/final_training/distilroberta/distilroberta_model.onnx \
    --checkpoint outputs/final_training/distilroberta/checkpoint \
    --reload \
    --log-level DEBUG

# Health check
curl http://127.0.0.1:8000/health

# Interactive docs
# Open: http://127.0.0.1:8000/docs
```

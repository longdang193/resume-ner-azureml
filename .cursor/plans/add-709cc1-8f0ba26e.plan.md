<!-- 8f0ba26e-4c87-4d17-8bad-6623e66613eb c47637a1-9626-450d-8e31-ecbbab109635 -->
# FastAPI Service Testing Notebook Implementation Plan

## Overview

Create a Jupyter notebook (`notebooks/02_test_fastapi_service.ipynb`) that systematically tests the FastAPI NER service. The notebook follows clean code principles: each cell performs a single task, uses meaningful names, avoids hard-coded values, and includes proper documentation.

## Clean Code Principles Applied

- **R1 (Avoid Hard-Coded Numbers)**: All configuration values (ports, timeouts, file paths) defined as named constants
- **R2 (Meaningful Names)**: Descriptive variable names that explain purpose without comments
- **R3 (Comments Sparingly)**: Code is self-explanatory; comments only for "why" not "what"
- **R4 (Single Responsibility)**: Each cell performs exactly one task
- **R5 (Document Complex Interfaces)**: Helper functions have docstrings with Args/Returns/Raises
- **R6 (DRY)**: Shared logic extracted into reusable functions
- **R7 (Standards)**: Python snake_case naming convention
- **R8 (Encapsulate Conditionals)**: Complex conditionals moved to named functions
- **R9 (Refactor Continuously)**: Code structure allows easy improvements
- **R10 (Version Control)**: Notebook structured for clear commit history

## Notebook Structure

### Cell 0: Title and Overview (Markdown)

- Notebook title and purpose
- Overview of testing steps
- Prerequisites and requirements

### Cell 1: Environment Detection (Code)

**Single Task**: Detect execution environment (Colab/Kaggle/Local)

- Detect environment using `os.environ` checks
- Set `BASE_DIR` constant based on environment
- Set `ROOT_DIR` constant for repository root
- No hard-coded paths; all paths derived from constants

### Cell 2: Repository Verification (Code)

**Single Task**: Verify repository structure exists

- Check `ROOT_DIR` exists
- Verify required directories: `["src", "config", "notebooks", "outputs"]`
- Raise clear `FileNotFoundError` if missing
- Print verification status

### Cell 3: Model File Discovery (Code)

**Single Task**: Locate ONNX model and checkpoint files

- Define search patterns as constants
- Search `outputs/` directory for ONNX files
- Search for checkpoint directories
- Store paths in named variables: `onnx_model_path`, `checkpoint_dir`
- Validate files exist with clear error messages

### Cell 4: Configuration Constants (Code)

**Single Task**: Define all configuration constants

- `API_HOST = "127.0.0.1"`
- `API_PORT = 8000`
- `API_URL = f"http://{API_HOST}:{API_PORT}"`
- `SERVER_START_TIMEOUT_SECONDS = 30`
- `REQUEST_TIMEOUT_SECONDS = 60`
- `PERFORMANCE_TEST_ITERATIONS = 10`
- `MAX_BATCH_SIZE_FOR_TESTING = 32`
- All magic numbers replaced with named constants

### Cell 5: Dependency Installation (Code)

**Single Task**: Install required Python packages

- Check if packages already installed
- Install only if missing: `fastapi`, `uvicorn`, `requests`, `python-multipart`
- Use try/except pattern to avoid redundant installations
- Print installation status

### Cell 6: Helper Function: Start Server (Code)

**Single Task**: Define function to start FastAPI server

```python
def start_fastapi_server(onnx_model_path, checkpoint_dir, host, port):
    """
    Start FastAPI server as background process.
    
    Args:
        onnx_model_path (Path): Path to ONNX model file.
        checkpoint_dir (Path): Path to checkpoint directory.
        host (str): Server host address.
        port (int): Server port number.
    
    Returns:
        subprocess.Popen: Server process object.
    
    Raises:
        FileNotFoundError: If run_api.py script not found.
    """
    # Implementation
```

- Uses meaningful parameter names
- Returns process object for later cleanup
- Handles errors clearly

### Cell 7: Helper Function: Wait for Server (Code)

**Single Task**: Define function to wait for server readiness

```python
def wait_for_server_ready(api_url, timeout_seconds):
    """
    Wait for API server to become ready.
    
    Args:
        api_url (str): Base URL of API server.
        timeout_seconds (int): Maximum seconds to wait.
    
    Returns:
        bool: True if server ready, False if timeout.
    """
    # Implementation
```

- Encapsulates polling logic
- Uses timeout constant from Cell 4
- Returns boolean for clear status

### Cell 8: Start Server (Code)

**Single Task**: Start the FastAPI server

- Call `start_fastapi_server()` with constants
- Call `wait_for_server_ready()` to verify
- Store process in `server_process` variable
- Print success/failure status
- Handle errors with clear messages

### Cell 9: Test Health Endpoint (Code)

**Single Task**: Test `/health` endpoint

- Make GET request to `f"{API_URL}/health"`
- Parse JSON response
- Verify `status` and `model_loaded` fields
- Print results clearly
- No nested conditionals; use early returns

### Cell 10: Test Info Endpoint (Code)

**Single Task**: Test `/info` endpoint

- Make GET request to `f"{API_URL}/info"`
- Extract and display model information
- Format output clearly
- Handle errors gracefully

### Cell 11: Sample Test Data (Code)

**Single Task**: Define sample test texts as constants

- `SAMPLE_RESUME_TEXT_1 = "..."` (meaningful name)
- `SAMPLE_RESUME_TEXT_2 = "..."`
- `SAMPLE_RESUME_TEXT_3 = "..."`
- Store in list: `SAMPLE_TEXTS = [...]`
- Each text represents different entity types

### Cell 12: Helper Function: Test Single Prediction (Code)

**Single Task**: Define reusable prediction test function

```python
def test_single_prediction(api_url, text, timeout_seconds):
    """
    Test single text prediction endpoint.
    
    Args:
        api_url (str): Base URL of API.
        text (str): Input text to predict.
        timeout_seconds (int): Request timeout.
    
    Returns:
        dict: Response JSON or None if error.
    
    Raises:
        requests.RequestException: If request fails.
    """
    # Implementation
```

- DRY principle: reusable for multiple tests
- Clear parameter names
- Returns structured data

### Cell 13: Test Single Prediction Endpoint (Code)

**Single Task**: Test `/predict` with sample text

- Call `test_single_prediction()` with first sample
- Display entities found
- Show processing time
- Format output clearly

### Cell 14: Helper Function: Format Entity Display (Code)

**Single Task**: Define function to format entity output

```python
def format_entity_display(entities):
    """
    Format entities for readable display.
    
    Args:
        entities (list): List of entity dictionaries.
    
    Returns:
        str: Formatted string representation.
    """
    # Implementation
```

- Encapsulates formatting logic
- Reusable across multiple cells

### Cell 15: Test Batch Prediction Endpoint (Code)

**Single Task**: Test `/predict/batch` endpoint

- Use `SAMPLE_TEXTS` constant
- Make POST request with all samples
- Display results per text
- Show total processing time
- Use `format_entity_display()` helper

### Cell 16: Test File Upload Preparation (Code)

**Single Task**: Locate test files if available

- Search for test PDF/image files
- Store paths in constants: `TEST_PDF_PATH`, `TEST_IMAGE_PATH`
- Check file existence
- Print availability status

### Cell 17: Helper Function: Test File Upload (Code)

**Single Task**: Define reusable file upload test function

```python
def test_file_upload(api_url, file_path, timeout_seconds):
    """
    Test file upload prediction endpoint.
    
    Args:
        api_url (str): Base URL of API.
        file_path (Path): Path to test file.
        timeout_seconds (int): Request timeout.
    
    Returns:
        dict: Response JSON or None if error.
    """
    # Implementation
```

- Handles file reading
- Constructs multipart form data
- Returns structured response

### Cell 18: Test File Upload Endpoint (Code)

**Single Task**: Test `/predict/file` with PDF (if available)

- Check if `TEST_PDF_PATH` exists
- Call `test_file_upload()` if available
- Display extracted text length
- Show entities found
- Skip gracefully if file not found

### Cell 19: Visualization Setup (Code)

**Single Task**: Import visualization libraries

- Import `pandas`, `matplotlib.pyplot`
- Set plotting style if needed
- No other logic in this cell

### Cell 20: Collect Prediction Results (Code)

**Single Task**: Collect results for visualization

- Run batch prediction
- Extract entities from all predictions
- Store in structured format (list of dicts)
- Prepare data for visualization

### Cell 21: Create Entity Statistics (Code)

**Single Task**: Calculate entity type statistics

- Count entities by type
- Create summary statistics
- Store in `entity_statistics` variable
- Print summary table

### Cell 22: Visualize Entity Distribution (Code)

**Single Task**: Create bar chart of entity types

- Use `entity_statistics` from previous cell
- Create bar plot
- Label axes clearly
- Set title meaningfully
- No data processing in this cell

### Cell 23: Visualize Confidence Distribution (Code)

**Single Task**: Create histogram of confidence scores

- Extract confidence values
- Create histogram
- Label axes clearly
- Only visualization logic

### Cell 24: Performance Test Setup (Code)

**Single Task**: Define performance test parameters

- `PERFORMANCE_TEST_TEXT = SAMPLE_TEXTS[0]`
- `PERFORMANCE_ITERATIONS = 10` (from constants)
- Store test text in named variable

### Cell 25: Helper Function: Measure Latency (Code)

**Single Task**: Define function to measure request latency

```python
def measure_prediction_latency(api_url, text, timeout_seconds):
    """
    Measure single prediction request latency.
    
    Args:
        api_url (str): Base URL of API.
        text (str): Test text.
        timeout_seconds (int): Request timeout.
    
    Returns:
        float: Latency in milliseconds.
    """
    # Implementation
```

- Encapsulates timing logic
- Returns milliseconds (named unit)

### Cell 26: Run Performance Tests (Code)

**Single Task**: Execute latency measurements

- Loop `PERFORMANCE_ITERATIONS` times
- Call `measure_prediction_latency()` each iteration
- Store results in `latency_results` list
- Calculate statistics using named functions

### Cell 27: Display Performance Statistics (Code)

**Single Task**: Calculate and display performance metrics

- Calculate mean, median, min, max, std dev
- Use `statistics` module functions
- Display formatted results
- No calculation logic; only display

### Cell 28: Helper Function: Test Batch Throughput (Code)

**Single Task**: Define function to test batch throughput

```python
def test_batch_throughput(api_url, batch_size, text, timeout_seconds):
    """
    Test batch prediction throughput.
    
    Args:
        api_url (str): Base URL of API.
        batch_size (int): Number of texts in batch.
        text (str): Sample text to repeat.
        timeout_seconds (int): Request timeout.
    
    Returns:
        tuple: (total_time_ms, throughput_texts_per_sec)
    """
    # Implementation
```

- Encapsulates throughput calculation
- Returns named tuple or dict

### Cell 29: Test Batch Throughput (Code)

**Single Task**: Test throughput with different batch sizes

- Define `BATCH_SIZES_TO_TEST = [1, 4, 8]` constant
- Loop through batch sizes
- Call `test_batch_throughput()` for each
- Display results in table format

### Cell 30: Error Handling Tests Setup (Code)

**Single Task**: Define error test cases

- `EMPTY_REQUEST = {}`
- `INVALID_BATCH_SIZE = {"texts": ["text"] * 100}`
- Store test cases as named constants

### Cell 31: Test Error Handling (Code)

**Single Task**: Test API error responses

- Test empty request
- Test batch size exceeded
- Test invalid endpoint
- Verify error response format
- Use helper function if repeated logic

### Cell 32: Interactive Testing Cell (Code)

**Single Task**: Provide interactive testing interface

- Define `CUSTOM_TEST_TEXT` variable (user can modify)
- Call prediction function
- Display formatted results
- Clear instructions in markdown above

### Cell 33: Cleanup Helper Function (Code)

**Single Task**: Define server cleanup function

```python
def stop_server(server_process, timeout_seconds):
    """
    Stop FastAPI server process gracefully.
    
    Args:
        server_process (subprocess.Popen): Server process.
        timeout_seconds (int): Timeout for graceful shutdown.
    
    Returns:
        bool: True if stopped successfully.
    """
    # Implementation
```

- Encapsulates cleanup logic
- Handles graceful vs forced shutdown

### Cell 34: Cleanup (Code)

**Single Task**: Stop server and cleanup

- Call `stop_server()` with `server_process`
- Print cleanup status
- Final summary message

## Implementation Details

### Constants to Define (R1 - Avoid Hard-Coded Numbers)

All configuration values as named constants:

- `DEFAULT_API_HOST = "127.0.0.1"`
- `DEFAULT_API_PORT = 8000`
- `SERVER_START_TIMEOUT_SECONDS = 30`
- `REQUEST_TIMEOUT_SECONDS = 60`
- `PERFORMANCE_TEST_ITERATIONS = 10`
- `MAX_BATCH_SIZE_FOR_TESTING = 32`
- `BATCH_SIZES_TO_TEST = [1, 4, 8]`
- `REQUIRED_DIRECTORIES = ["src", "config", "notebooks", "outputs"]`

### Helper Functions (R4, R5, R6)

All reusable logic extracted into functions with docstrings:

- `start_fastapi_server()` - Server startup
- `wait_for_server_ready()` - Server readiness check
- `test_single_prediction()` - Single prediction test
- `test_file_upload()` - File upload test
- `format_entity_display()` - Entity formatting
- `measure_prediction_latency()` - Latency measurement
- `test_batch_throughput()` - Throughput testing
- `stop_server()` - Server cleanup

### Naming Conventions (R2, R7)

- All variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Functions: `snake_case` with verb-noun pattern
- Clear, descriptive names that explain purpose

### Cell Organization (R4)

- Each cell performs exactly one task
- Related cells grouped with markdown headers
- Clear separation: setup → testing → visualization → cleanup
- No cell does multiple unrelated things

### Error Handling (R8)

- Complex conditionals moved to helper functions
- Early returns for error cases
- Clear error messages with context
- Graceful degradation (e.g., skip file tests if files missing)

## File Structure

**File**: `notebooks/02_test_fastapi_service.ipynb`

- Markdown cells for documentation
- Code cells following single-responsibility principle
- Helper functions defined before use
- Constants defined early
- Clear progression from setup to testing to cleanup

## Success Criteria

- Notebook runs end-to-end without errors
- All cells follow single-responsibility principle
- No hard-coded magic numbers
- All functions have docstrings
- Code is self-documenting (minimal comments)
- Error handling is graceful
- Results are clearly displayed
- Server cleanup is reliable

### To-dos

- [ ] Create API package structure (__init__.py, app.py, config.py)
- [ ] Create Pydantic models for requests and responses
- [ ] Implement ONNX model loading and inference engine
- [ ] Implement token-to-entity span conversion and decoding
- [ ] Create /predict and /predict/batch endpoints for text input
- [ ] Implement PDF text extraction and OCR for images
- [ ] Create /predict/file endpoints for PDF and image uploads
- [ ] Add custom exceptions, validation, and error handling middleware
- [ ] Create /health and /info endpoints for monitoring
- [ ] Create run_api.py CLI script for starting the server
- [ ] Write unit and integration tests for API endpoints
- [ ] Create API_USAGE.md and DEPLOYMENT.md documentation
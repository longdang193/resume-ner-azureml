## Document Information

| Field | Value |
| ----------------------- | ----------------------------------------------- |
| System Under Test (SUT) | Local FastAPI Inference Server |
| Test Type | Functional, Integration, Performance, Stability |
| Environment | Local development, CI |
| Interfaces | REST (HTTP/JSON, multipart/form-data) |
| Document Version | 1.0 |
| Status | Draft |

## Purpose

This document defines the quality assurance (QA) and test specification for the local FastAPI inference server.

The goal is to ensure the system is functionally correct, robust to invalid inputs, stable under repeated workloads, and performant under expected usage patterns.

This specification is designed to:

* Detect functional and integration defects early
* Prevent startup hangs and server instability
* Validate performance consistency across repeated runs
* Support automated testing in CI pipelines

## Scope

### In Scope

* Server lifecycle and startup behavior
* Model loading and readiness detection
* REST API endpoints for prediction and diagnostics
* File upload and batch processing workflows
* Error handling and edge cases
* Performance and consistency validation
* End-to-end (E2E) integration scenarios

### Out of Scope

* Model training correctness
* Model accuracy benchmarking beyond deterministic checks
* Production deployment (cloud, container orchestration)

## Test Strategy

Testing is structured into multiple layers:

| Test Layer | Description |
| ----------------------------- | ------------------------------------------------ |
| Server Lifecycle Tests | Validate startup, shutdown, and failure handling |
| API Functional Tests | Validate correctness of individual endpoints |
| Integration Tests | Validate multi-step workflows |
| Performance Tests | Measure latency, throughput, and resource usage |
| Stability & Consistency Tests | Detect nondeterminism and regressions |

## Test Environment

### Software

* Python (project-supported version)
* FastAPI
* HTTP client (e.g., httpx, requests)
* OCR and PDF parsing dependencies (as applicable)

### Configuration

* Valid and invalid model paths
* Configurable port
* Environment variables and runtime flags

## Test Categories & Specifications

### Server Lifecycle & Startup Tests

**Objective:** Ensure reliable startup, shutdown, and failure behavior.

| Test Case | Expected Result |
| --------------------------------------- | ---------------------------------------------- |
| Startup with valid model path | Server starts successfully; model loaded |
| Startup with invalid/missing model path | Server fails fast with clear error |
| Port already in use | Server fails with explicit port conflict error |
| Model loading verification | Server reports ready only after model load |
| Graceful shutdown | Resources released; no hanging processes |
| **Startup hang prevention** | Startup must timeout if readiness not reached |

**Acceptance Criteria**

* No startup scenario may hang indefinitely
* Port checks and wait loops must have bounded timeouts

### Health & Info Endpoints

**Objective:** Validate server readiness signaling.

| Endpoint | Condition | Expected Status |
| ------------- | ---------------- | ------------------------- |
| `GET /health` | Model loaded | `200 OK` |
| `GET /health` | Model not loaded | `503 Service Unavailable` |
| `GET /info` | Model loaded | `200 OK` with metadata |
| `GET /info` | Model not loaded | `503 Service Unavailable` |

### Single Text Prediction (`POST /predict`)

**Objective:** Validate single-text inference correctness and robustness.

**Test Scenarios**

* **Valid inputs:**
  * Normal resume text with entities
  * Text with various entity types
  * Text with special characters (emails, URLs, phone numbers)
* **Edge cases:**
  * Empty string
  * Very long text (10,000+ characters)
  * Unicode characters (non-ASCII)
  * Whitespace-only text
* **Error cases:**
  * Missing `text` field
  * Invalid JSON structure
  * Non-string text value

**Acceptance Criteria**

* Valid requests return predictions with entities
* Invalid inputs return clear, structured errors (422 Unprocessable Entity)
* Response schema includes required fields (entities, processing_time_ms)
* Edge cases are handled gracefully (may return empty entities or validation errors)

### Batch Text Prediction (`POST /predict/batch`)

**Objective:** Validate batch inference behavior and constraints.

**Test Scenarios**

* **Valid inputs:**
  * Small batch (2-3 texts)
  * Medium batch (5-10 texts)
  * Large batch (up to MAX_BATCH_SIZE, typically 32)
* **Edge cases:**
  * Empty batch list
  * Batch with mixed valid/invalid texts
  * Batch with one empty text
* **Error cases:**
  * Batch exceeding MAX_BATCH_SIZE (32)
  * Missing `texts` field
  * Non-list value for texts

**Acceptance Criteria**

* Batch limits are enforced (returns 400 or 422 when exceeded)
* Partial failures are handled consistently (may return partial results)
* Response format is stable and documented (predictions array, total_processing_time_ms)
* Empty batch returns validation error (422)

### File Upload Prediction (`POST /predict/file`)

**Objective:** Validate file ingestion and inference pipeline.

**Test Scenarios**

* **Valid inputs:**
  * PDF file upload
  * PNG image file upload (if OCR dependencies available)
  * Larger PDF files
* **Edge cases:**
  * Small PDF files
* **Error cases:**
  * Invalid file type (e.g., .txt file)
  * Missing file in request

**Acceptance Criteria**

* Valid files return predictions and extracted text
* Invalid files return explicit error messages (400 Bad Request)
* File size limits are enforced (MAX_FILE_SIZE, typically 10MB)
* Missing file returns validation error (422)
* OCR dependencies gracefully handled (skip or error message if not installed)

### Batch File Upload (`POST /predict/file/batch`)

**Objective:** Validate batch file processing.

**Test Scenarios**

* **Valid inputs:**
  * Small batch (2-3 files)
  * Medium batch (5 files)
  * Mixed file types (PDF and PNG)
* **Edge cases:**
  * Empty batch
* **Error cases:**
  * Batch exceeding MAX_BATCH_SIZE

**Acceptance Criteria**

* Multiple files processed correctly
* Mixed file types handled gracefully
* Partial failures reported clearly
* Batch size limits enforced (400 Bad Request when exceeded)
* Empty batch returns validation error (422)

### Debug Endpoint (`POST /predict/debug`)

**Objective:** Validate diagnostic and debug output.

**Expected Output**

* Token-level predictions
* Offset mappings
* Confidence scores
* Detailed intermediate information

### Error Handling & Edge Cases

**Objective:** Ensure predictable and safe failure behavior.

| Scenario | Expected Status |
| ----------------------- | -------------------------- |
| Invalid JSON | `422` |
| Missing required fields | `422` |
| Model not loaded | `503` |
| Unexpected server error | Graceful error response |
| Long-running request | Timeout handled explicitly |

## Performance & Stability Tests

### Performance Tests

**Metrics**

* Single-request latency
* Batch throughput
* Concurrent request handling (if supported)
* Memory usage during batch processing

### Performance Consistency Tests

**Objective:** Detect nondeterminism and regressions.

**Test Procedure**

* Repeatedly process multiple files of similar size and content
* Run the same workload multiple times

**Acceptance Criteria**

* No request timeouts
* Latency variance remains within defined thresholds
* Outputs remain consistent across runs
* No performance degradation over repeated executions

## Integration & End-to-End Tests

**Objective:** Validate full system behavior.

**Scenarios**

* File upload → extraction → prediction → response
* Multiple sequential requests
* Server restart followed by model reload
* Configuration validation across restarts

## Test Automation & CI Requirements

* All tests must be automatable
* Tests must be deterministic and fast-running
* Clear failure messages are required
* No test may rely on indefinite waits or sleeps
* Tests should be runnable locally and in CI

## Exit Criteria

The system is considered **QA-approved** when:

* All critical and high-severity tests pass
* No startup hangs are observed
* Performance consistency tests meet defined thresholds
* No flaky or nondeterministic tests remain

## Risks & Mitigations

| Risk | Mitigation |
| ----------------------- | ---------------------------- |
| Startup hangs | Enforce startup timeouts |
| Flaky performance tests | Use deterministic inputs |
| OCR variability | Use controlled test fixtures |

## Appendix

This document references the following appendices:

* [Test Data Descriptions](Appendices/test_data_descriptions.md) - Detailed specifications of text fixtures, file fixtures, batch fixtures, and determinism requirements
* [Performance Thresholds](Appendices/performance_thresholds.md) - Latency, throughput, startup, and consistency thresholds with reporting requirements

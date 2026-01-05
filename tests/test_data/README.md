# Test Data for FastAPI Local Server Tests

This directory contains test data fixtures used for testing the FastAPI inference server.

## Structure

- **PDF files**: `test_resume_ner_*.pdf` - PDF versions of test resumes
- **PNG files**: `test_resume_ner_*.png` - PNG image versions of test resumes (identical content)
- **Larger test files**: `test_resume*.pdf` and `test_resume*.png` - Larger test files for performance testing
- **`fixtures.py`**: Python module providing programmatic access to test data
- **`generate_test_files.py`**: Script to generate PDF and PNG files from text

## Usage

### Programmatic Access

Use the `fixtures.py` module to access test data in tests:

```python
from tests.test_data.fixtures import (
    get_text_fixture,
    get_file_fixture,
    get_batch_text_fixture,
    get_batch_file_fixture,
)

# Get text fixture
text = get_text_fixture("text_1")

# Get file fixture
pdf_path = get_file_fixture("file_1", "pdf")
png_path = get_file_fixture("file_1", "png")

# Get batch fixtures
texts = get_batch_text_fixture("batch_text_small")
files = get_batch_file_fixture("batch_file_small", "pdf")
```

### Available Fixtures

#### Text Fixtures

- `text_1` through `text_10`: Standard resume text samples
- `text_empty`: Empty string
- `text_unicode`: Unicode characters
- `text_long`: Very long text (10,000+ characters)
- `text_special`: Special characters (email, phone, URL)

#### File Fixtures

- `file_1` through `file_10`: Standard test files (PDF and PNG)
- `file_resume_1`, `file_resume_2`, `file_resume_3`: Larger test files

#### Batch Fixtures

- `batch_text_small`: 3 text items
- `batch_text_medium`: 5 text items
- `batch_text_large`: 10 text items
- `batch_text_empty`: Empty batch
- `batch_text_mixed`: Mixed valid/invalid items
- `batch_file_small`: 3 files
- `batch_file_medium`: 5 files
- `batch_file_large`: 10 files
- `batch_file_mixed_types`: Mixed PDF/PNG files

## Generating Test Files

If test files are missing, generate them using:

```bash
cd tests/test_data
python generate_test_files.py
```

This requires:
- `reportlab` or `fpdf2` for PDF generation
- `Pillow` for PNG generation

Install dependencies:

```bash
pip install -r requirements_generate.txt
```

## File Content

Each pair of PDF/PNG files (e.g., `test_resume_ner_1.pdf` and `test_resume_ner_1.png`) contains identical text content. This allows testing consistency across different file format handling.

The text content matches the corresponding text fixtures in `fixtures.py`:
- `test_resume_ner_1.*` contains `text_1` content
- `test_resume_ner_2.*` contains `text_2` content
- etc.

## Validation

To validate that all required test files exist:

```python
from tests.test_data.fixtures import validate_all_fixtures

result = validate_all_fixtures()
print(f"Missing: {result['missing']}")
print(f"Found: {len(result['found'])} files")
```

## Determinism Requirements

- All test files are version-controlled
- Files are generated deterministically from known text content
- Same input produces same output (within floating-point tolerance)
- No external network dependencies required

## References

See `docs/qa/QA_Local_FastAPI_Inference_Server/Appendices/test_data_descriptions.md` for detailed specifications of all test data fixtures.


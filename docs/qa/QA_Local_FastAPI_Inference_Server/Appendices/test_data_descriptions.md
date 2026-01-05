# Test Data Descriptions

Test data must be deterministic, versioned, and representative of real usage while remaining lightweight.

## Text Fixtures

Text fixtures are used for testing the `/predict` and `/predict/batch` endpoints. The following text samples are extracted from the test file content and represent typical resume text with various NER entities:

| ID | Description | Content Preview | Entity Types |
| -- | ----------- | --------------- | ------------ |
| `text_1` | AWS ML Engineer | "Amazon Web Services, Austin, TX. Lead Machine Learning Engineer with 7 years of experience in NLP, recommender systems, and deep learning. Skills: Python, PyTorch, Spark." | ORG, LOC, TITLE, SKILL |
| `text_2` | Meta Data Analyst | "Alice Johnson is a data analyst at Meta Platforms. Email: <alice.johnson@meta.com>. Phone: +1-408-555-7890. Location: Menlo Park, CA." | PERSON, ORG, EMAIL, PHONE, LOC |
| `text_3` | Stanford PhD Researcher | "Robert Lee holds a PhD in Artificial Intelligence from Stanford University. His expertise includes computer vision, transformers, and TensorFlow." | PERSON, ORG, SKILL |
| `text_4` | Netflix Software Engineer | "Netflix Inc., Los Gatos, CA. Senior Software Engineer with 6 years of experience in backend systems and machine learning. Skills: Java, Scala, Kubernetes." | ORG, LOC, TITLE, SKILL |
| `text_5` | OpenAI Research Scientist | "Emily Davis is a research scientist at OpenAI. Email: <emily.davis@openai.com>. Phone: +1-212-555-3344. Location: San Francisco, CA." | PERSON, ORG, EMAIL, PHONE, LOC |
| `text_6` | University of Toronto Graduate | "Michael Brown earned a Bachelor's degree in Data Science from the University of Toronto. He focuses on statistical modeling, SQL, and Python." | PERSON, ORG, SKILL |
| `text_7` | Apple AI Engineer | "Apple Inc., Cupertino, CA. Principal AI Engineer with 10 years of experience in speech recognition and large-scale ML systems. Skills: C++, PyTorch, ONNX." | ORG, LOC, TITLE, SKILL |
| `text_8` | Shopify Developer | "Daniel Kim is a full-stack developer at Shopify. Email: <daniel.kim@shopify.com>. Phone: +1-647-555-9981. Location: Toronto, ON." | PERSON, ORG, EMAIL, PHONE, LOC |
| `text_9` | CMU Software Engineer | "Sarah Wilson has a Master of Science in Software Engineering from Carnegie Mellon University. She specializes in distributed systems and cloud computing." | PERSON, ORG, SKILL |
| `text_10` | IBM Research Scientist | "IBM Research, Yorktown Heights, NY. Applied Scientist with 8 years of experience in natural language understanding and information extraction. Skills: Python, spaCy, Hugging Face." | ORG, LOC, TITLE, SKILL |

### Edge Case Text Fixtures

| ID | Description | Content |
| -- | ----------- | ------- |
| `text_empty` | Empty string | `""` |
| `text_unicode` | Unicode characters | `"José García, 北京, 東京, München"` |
| `text_long` | Very long text | 10,000+ character string (repeated text_1 content) |
| `text_special` | Special characters | `"Email: test@example.com, Phone: +1-555-123-4567, URL: https://example.com"` |

## File Fixtures

File fixtures are located in `tests/test_data/` and are used for testing the `/predict/file` and `/predict/file/batch` endpoints. Each file has both PDF and PNG versions with identical content.

**Note:** Files with the same name but different file types (e.g., `test_resume_ner_1.pdf` and `test_resume_ner_1.png`) contain identical text content, allowing for testing consistency across file format handling.

### Primary Test Files (test_resume_ner_1 through test_resume_ner_10)

| File ID | PDF Path | PNG Path | Content | Size (approx) | Use Case |
| ------- | -------- | -------- | ------- | ------------- | -------- |
| `file_1` | `test_resume_ner_1.pdf` | `test_resume_ner_1.png` | AWS ML Engineer resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_2` | `test_resume_ner_2.pdf` | `test_resume_ner_2.png` | Meta Data Analyst resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_3` | `test_resume_ner_3.pdf` | `test_resume_ner_3.png` | Stanford PhD Researcher resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_4` | `test_resume_ner_4.pdf` | `test_resume_ner_4.png` | Netflix Software Engineer resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_5` | `test_resume_ner_5.pdf` | `test_resume_ner_5.png` | OpenAI Research Scientist resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_6` | `test_resume_ner_6.pdf` | `test_resume_ner_6.png` | University of Toronto Graduate resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_7` | `test_resume_ner_7.pdf` | `test_resume_ner_7.png` | Apple AI Engineer resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_8` | `test_resume_ner_8.pdf` | `test_resume_ner_8.png` | Shopify Developer resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_9` | `test_resume_ner_9.pdf` | `test_resume_ner_9.png` | CMU Software Engineer resume text | ~1.1 KB | Standard PDF/PNG test |
| `file_10` | `test_resume_ner_10.pdf` | `test_resume_ner_10.png` | IBM Research Scientist resume text | ~1.1 KB | Standard PDF/PNG test |

### Additional Test Files

| File ID | PDF Path | PNG Path | Size (approx) | Use Case |
| ------- | -------- | -------- | ------------- | -------- |
| `file_resume_1` | `test_resume.pdf` | `test_resume.png` | ~56 KB | Larger PDF test |
| `file_resume_2` | `test_resume_2.pdf` | `test_resume_2.png` | ~41 KB | Medium PDF test |
| `file_resume_3` | `test_resume_3.pdf` | `test_resume_3.png` | ~46 KB | Medium PDF test |

### Invalid File Fixtures (to be created or mocked)

| File ID | Description | Use Case |
| ------- | ----------- | -------- |
| `file_invalid_type` | File with unsupported extension (e.g., `.txt`, `.docx`) | Invalid file type rejection |
| `file_corrupted_pdf` | Corrupted PDF file | Error handling for malformed files |
| `file_corrupted_png` | Corrupted PNG file | Error handling for malformed images |
| `file_oversized` | File exceeding size limits | File size limit enforcement |

## Batch Fixtures

Batch fixtures are composed of combinations of the text and file fixtures above.

### Text Batch Fixtures

| Batch ID | Description | Contents | Size |
| -------- | ----------- | -------- | ---- |
| `batch_text_small` | Small batch | `[text_1, text_2, text_3]` | 3 items |
| `batch_text_medium` | Medium batch | `[text_1, ..., text_5]` | 5 items |
| `batch_text_large` | Large batch | `[text_1, ..., text_10]` | 10 items |
| `batch_text_empty` | Empty batch | `[]` | 0 items |
| `batch_text_mixed` | Mixed valid/invalid | `[text_1, text_empty, text_3]` | 3 items (1 invalid) |
| `batch_text_max` | Maximum batch size | `[text_1, ..., text_N]` where N = `MAX_BATCH_SIZE` | N items |

### File Batch Fixtures

| Batch ID | Description | Contents | Size |
| -------- | ----------- | -------- | ---- |
| `batch_file_small` | Small file batch | `[file_1.pdf, file_2.pdf, file_3.pdf]` | 3 files |
| `batch_file_medium` | Medium file batch | `[file_1.pdf, ..., file_5.pdf]` | 5 files |
| `batch_file_large` | Large file batch | `[file_1.pdf, ..., file_10.pdf]` | 10 files |
| `batch_file_mixed_types` | Mixed PDF/PNG | `[file_1.pdf, file_2.png, file_3.pdf]` | 3 files (mixed) |
| `batch_file_mixed_validity` | Mixed valid/invalid | `[file_1.pdf, file_invalid_type, file_3.pdf]` | 3 files (1 invalid) |
| `batch_file_empty` | Empty batch | `[]` | 0 files |
| `batch_file_max` | Maximum batch size | `[file_1.pdf, ..., file_N.pdf]` where N = `MAX_BATCH_SIZE` | N files |

## Determinism Requirements

* **Fixed random seeds:** All random number generators (if any) must use fixed seeds in test environments
* **Controlled OCR inputs:** File fixtures are generated deterministically from known text content
* **No external network dependencies:** Tests must not require internet connectivity or external services
* **Identical inputs produce identical outputs:** Same input must produce same output (or within defined tolerance for floating-point comparisons)
* **Versioned fixtures:** Test data files are version-controlled and should not be modified without updating test expectations
* **Reproducible file generation:** The `generate_test_files.py` script ensures consistent file generation across environments

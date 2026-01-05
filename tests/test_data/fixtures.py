"""Test data fixtures for FastAPI local server tests.

This module provides deterministic test data fixtures (text, files, batches)
as specified in docs/qa/QA_Local_FastAPI_Inference_Server/Appendices/test_data_descriptions.md
"""

from pathlib import Path
from typing import List, Dict, Optional

# Base directory for test data files
TEST_DATA_DIR = Path(__file__).parent

# Text fixtures - matching test_data_descriptions.md
TEXT_FIXTURES: Dict[str, str] = {
    "text_1": "Amazon Web Services, Austin, TX. Lead Machine Learning Engineer with 7 years of experience in NLP, recommender systems, and deep learning. Skills: Python, PyTorch, Spark.",
    "text_2": "Alice Johnson is a data analyst at Meta Platforms. Email: alice.johnson@meta.com. Phone: +1-408-555-7890. Location: Menlo Park, CA.",
    "text_3": "Robert Lee holds a PhD in Artificial Intelligence from Stanford University. His expertise includes computer vision, transformers, and TensorFlow.",
    "text_4": "Netflix Inc., Los Gatos, CA. Senior Software Engineer with 6 years of experience in backend systems and machine learning. Skills: Java, Scala, Kubernetes.",
    "text_5": "Emily Davis is a research scientist at OpenAI. Email: emily.davis@openai.com. Phone: +1-212-555-3344. Location: San Francisco, CA.",
    "text_6": "Michael Brown earned a Bachelor's degree in Data Science from the University of Toronto. He focuses on statistical modeling, SQL, and Python.",
    "text_7": "Apple Inc., Cupertino, CA. Principal AI Engineer with 10 years of experience in speech recognition and large-scale ML systems. Skills: C++, PyTorch, ONNX.",
    "text_8": "Daniel Kim is a full-stack developer at Shopify. Email: daniel.kim@shopify.com. Phone: +1-647-555-9981. Location: Toronto, ON.",
    "text_9": "Sarah Wilson has a Master of Science in Software Engineering from Carnegie Mellon University. She specializes in distributed systems and cloud computing.",
    "text_10": "IBM Research, Yorktown Heights, NY. Applied Scientist with 8 years of experience in natural language understanding and information extraction. Skills: Python, spaCy, Hugging Face.",
    # Edge cases
    "text_empty": "",
    "text_unicode": "José García, 北京, 東京, München",
    "text_long": "",  # Will be generated dynamically
    "text_special": "Email: test@example.com, Phone: +1-555-123-4567, URL: https://example.com",
}

# Generate long text (10,000+ characters)
if not TEXT_FIXTURES["text_long"]:
    base_text = TEXT_FIXTURES["text_1"]
    repetitions = (10000 // len(base_text)) + 1
    TEXT_FIXTURES["text_long"] = (base_text + " ") * repetitions

# File fixture mappings
FILE_FIXTURES: Dict[str, Dict[str, str]] = {
    "file_1": {"pdf": "test_resume_ner_1.pdf", "png": "test_resume_ner_1.png"},
    "file_2": {"pdf": "test_resume_ner_2.pdf", "png": "test_resume_ner_2.png"},
    "file_3": {"pdf": "test_resume_ner_3.pdf", "png": "test_resume_ner_3.png"},
    "file_4": {"pdf": "test_resume_ner_4.pdf", "png": "test_resume_ner_4.png"},
    "file_5": {"pdf": "test_resume_ner_5.pdf", "png": "test_resume_ner_5.png"},
    "file_6": {"pdf": "test_resume_ner_6.pdf", "png": "test_resume_ner_6.png"},
    "file_7": {"pdf": "test_resume_ner_7.pdf", "png": "test_resume_ner_7.png"},
    "file_8": {"pdf": "test_resume_ner_8.pdf", "png": "test_resume_ner_8.png"},
    "file_9": {"pdf": "test_resume_ner_9.pdf", "png": "test_resume_ner_9.png"},
    "file_10": {"pdf": "test_resume_ner_10.pdf", "png": "test_resume_ner_10.png"},
    "file_resume_1": {"pdf": "test_resume.pdf", "png": "test_resume.png"},
    "file_resume_2": {"pdf": "test_resume_2.pdf", "png": "test_resume_2.png"},
    "file_resume_3": {"pdf": "test_resume_3.pdf", "png": "test_resume_3.png"},
}


def get_text_fixture(fixture_id: str) -> str:
    """
    Get text fixture by ID.
    
    Args:
        fixture_id: Fixture identifier (e.g., "text_1", "text_empty", "text_long")
    
    Returns:
        Text string
    
    Raises:
        KeyError: If fixture_id not found
    """
    if fixture_id not in TEXT_FIXTURES:
        raise KeyError(
            f"Text fixture '{fixture_id}' not found. "
            f"Available fixtures: {list(TEXT_FIXTURES.keys())}"
        )
    return TEXT_FIXTURES[fixture_id]


def get_file_fixture(fixture_id: str, file_type: str = "pdf") -> Path:
    """
    Get file fixture path.
    
    Args:
        fixture_id: Fixture identifier (e.g., "file_1", "file_resume_1")
        file_type: File type ("pdf" or "png")
    
    Returns:
        Path to the file fixture
    
    Raises:
        KeyError: If fixture_id not found
        ValueError: If file_type is invalid
        FileNotFoundError: If file does not exist
    """
    if file_type not in ("pdf", "png"):
        raise ValueError(f"Invalid file_type: {file_type}. Must be 'pdf' or 'png'")
    
    if fixture_id not in FILE_FIXTURES:
        raise KeyError(
            f"File fixture '{fixture_id}' not found. "
            f"Available fixtures: {list(FILE_FIXTURES.keys())}"
        )
    
    filename = FILE_FIXTURES[fixture_id].get(file_type)
    if not filename:
        raise KeyError(f"File type '{file_type}' not available for fixture '{fixture_id}'")
    
    file_path = TEST_DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {file_path}. "
            f"Run tests/test_data/generate_test_files.py to create test files."
        )
    
    return file_path


def get_batch_text_fixture(batch_id: str) -> List[str]:
    """
    Get batch text fixture.
    
    Args:
        batch_id: Batch identifier (e.g., "batch_text_small", "batch_text_medium")
    
    Returns:
        List of text strings
    
    Raises:
        KeyError: If batch_id not found
    """
    batch_definitions: Dict[str, List[str]] = {
        "batch_text_small": ["text_1", "text_2", "text_3"],
        "batch_text_medium": ["text_1", "text_2", "text_3", "text_4", "text_5"],
        "batch_text_large": [
            "text_1", "text_2", "text_3", "text_4", "text_5",
            "text_6", "text_7", "text_8", "text_9", "text_10"
        ],
        "batch_text_empty": [],
        "batch_text_mixed": ["text_1", "text_empty", "text_3"],
    }
    
    if batch_id not in batch_definitions:
        raise KeyError(
            f"Batch text fixture '{batch_id}' not found. "
            f"Available fixtures: {list(batch_definitions.keys())}"
        )
    
    text_ids = batch_definitions[batch_id]
    return [get_text_fixture(text_id) for text_id in text_ids]


def get_batch_file_fixture(
    batch_id: str,
    file_type: str = "pdf",
    max_batch_size: Optional[int] = None
) -> List[Path]:
    """
    Get batch file fixture.
    
    Args:
        batch_id: Batch identifier (e.g., "batch_file_small", "batch_file_medium")
        file_type: File type ("pdf" or "png")
        max_batch_size: Maximum batch size (for batch_file_max)
    
    Returns:
        List of file paths
    
    Raises:
        KeyError: If batch_id not found
        ValueError: If max_batch_size required but not provided
    """
    batch_definitions: Dict[str, List[str]] = {
        "batch_file_small": ["file_1", "file_2", "file_3"],
        "batch_file_medium": ["file_1", "file_2", "file_3", "file_4", "file_5"],
        "batch_file_large": [
            "file_1", "file_2", "file_3", "file_4", "file_5",
            "file_6", "file_7", "file_8", "file_9", "file_10"
        ],
        "batch_file_empty": [],
        "batch_file_mixed_types": ["file_1", "file_2", "file_3"],  # Will be mixed in code
    }
    
    if batch_id == "batch_file_max":
        if max_batch_size is None:
            raise ValueError("max_batch_size required for batch_file_max")
        # Generate list of file_1 through file_N
        file_ids = [f"file_{i}" for i in range(1, max_batch_size + 1)]
        return [get_file_fixture(fid, file_type) for fid in file_ids]
    
    if batch_id == "batch_file_mixed_types":
        # Return mixed PDF/PNG files
        return [
            get_file_fixture("file_1", "pdf"),
            get_file_fixture("file_2", "png"),
            get_file_fixture("file_3", "pdf"),
        ]
    
    if batch_id not in batch_definitions:
        raise KeyError(
            f"Batch file fixture '{batch_id}' not found. "
            f"Available fixtures: {list(batch_definitions.keys())} + 'batch_file_max'"
        )
    
    file_ids = batch_definitions[batch_id]
    return [get_file_fixture(fid, file_type) for fid in file_ids]


def validate_all_fixtures() -> Dict[str, List[str]]:
    """
    Validate that all required test files exist.
    
    Returns:
        Dictionary with 'missing' and 'found' keys containing lists of file paths
    """
    missing = []
    found = []
    
    # Check all file fixtures
    for fixture_id, file_map in FILE_FIXTURES.items():
        for file_type, filename in file_map.items():
            file_path = TEST_DATA_DIR / filename
            if file_path.exists():
                found.append(str(file_path))
            else:
                missing.append(str(file_path))
    
    return {"missing": missing, "found": found}


def get_all_text_fixture_ids() -> List[str]:
    """Get list of all available text fixture IDs."""
    return list(TEXT_FIXTURES.keys())


def get_all_file_fixture_ids() -> List[str]:
    """Get list of all available file fixture IDs."""
    return list(FILE_FIXTURES.keys())


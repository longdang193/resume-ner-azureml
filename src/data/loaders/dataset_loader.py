"""Data loading and processing utilities."""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    import torch


def load_dataset(data_path: str) -> Dict[str, Any]:
    """
    Load dataset from JSON files.

    Args:
        data_path: Path to directory containing train.json and optionally validation.json.

    Returns:
        Dictionary with "train" and "validation" keys containing data lists.

    Raises:
        FileNotFoundError: If dataset path or train.json does not exist.
    """
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    train_file = data_path_obj / "train.json"
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    val_file = data_path_obj / "validation.json"
    val_data = []
    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            val_data = json.load(f)

    test_file = data_path_obj / "test.json"
    test_data = []
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

    return {"train": train_data, "validation": val_data, "test": test_data}


def build_label_list(data_config: Dict[str, Any]) -> List[str]:
    """
    Build label list from data configuration.

    Args:
        data_config: Data configuration dictionary containing schema information.

    Returns:
        List of labels starting with "O" followed by sorted entity types.
    """
    entity_types = data_config.get("schema", {}).get("entity_types", [])
    return ["O"] + sorted(entity_types)


def _compute_entity_presence_labels(
    dataset: List[Dict[str, Any]],
    entity_types: Optional[List[str]] = None,
) -> List[tuple]:
    """
    Compute per-document entity presence labels for stratification.

    Each label is a tuple of entity types present in the document. This helps
    keep rare entity types represented in each split/fold.
    """
    labels: List[tuple] = []
    for sample in dataset:
        annotations = sample.get("annotations", []) or []
        present = []
        for ann in annotations:
            if not isinstance(ann, (list, tuple)) or len(ann) < 3:
                continue
            ent = ann[2]
            if entity_types and ent not in entity_types:
                continue
            present.append(ent)
        labels.append(tuple(sorted(set(present))))
    return labels


def split_train_test(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    stratified: bool = False,
    random_seed: int = 42,
    entity_types: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train and test sets, with optional stratification.

    Args:
        dataset: Full dataset samples.
        train_ratio: Proportion of data to use for training (0-1).
        stratified: If True, stratify by entity presence.
        random_seed: Random seed for reproducibility.
        entity_types: Optional entity type whitelist for stratification labels.

    Returns:
        (train_data, test_data)
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    test_size = 1.0 - train_ratio

    stratify_labels = None
    if stratified:
        entity_presence_tuples = _compute_entity_presence_labels(
            dataset, entity_types)
        # Convert tuples to strings for sklearn compatibility
        # Tuples can't be directly used as stratify labels because they have variable lengths
        stratify_labels = [",".join(sorted(present)) if present else "none"
                           for present in entity_presence_tuples]

        # Check if stratification is feasible
        # Sklearn requires at least 2 samples per class for stratified splitting
        from collections import Counter
        label_counts = Counter(stratify_labels)
        unique_classes = len(set(stratify_labels))
        min_class_count = min(label_counts.values()) if label_counts else 0

        # If all labels are identical or any class has < 2 samples, disable stratification
        if unique_classes <= 1 or min_class_count < 2:
            if stratified:  # Only warn if user explicitly requested stratification
                import warnings
                warnings.warn(
                    f"Stratified splitting not feasible: "
                    f"{unique_classes} unique classes, "
                    f"minimum class count: {min_class_count}. "
                    f"Falling back to non-stratified splitting.",
                    UserWarning
                )
            stratify_labels = None

    train_data, test_data = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    return list(train_data), list(test_data)


def save_split_files(
    output_dir: Path,
    train_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> None:
    """
    Persist train/test splits to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Serialize with defensive encoding to handle malformed surrogate pairs

    def _dump_safe(data: List[Dict[str, Any]], path: Path) -> None:
        # Convert to JSON string first
        content = json.dumps(data, ensure_ascii=False, indent=2)
        # Encode/decode with errors="replace" to strip/replace invalid surrogates
        content = content.encode(
            "utf-8", errors="replace").decode("utf-8", errors="replace")
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)

    _dump_safe(train_data, output_dir / "train.json")
    _dump_safe(test_data, output_dir / "test.json")


def normalize_text_for_tokenization(raw_text: Any) -> str:
    """
    Normalize arbitrary text input for tokenization.

    This function is training-specific and converts various input types (string, list,
    dict, etc.) into a single string that tokenizers can consume. It handles type
    conversion and flattening of nested structures.

    Note: This is distinct from naming/path normalization in core/normalize.py.
    This function is specifically for preparing text data for model tokenization.

    Args:
        raw_text: Input text in any format (string, list, dict, etc.)

    Returns:
        Normalized string suitable for tokenization. Always returns a string,
        never None, list, tuple, or other types. Empty string if input is None or empty.
    """
    # Handle None explicitly
    if raw_text is None:
        return ""

    # Handle string (most common case) - return immediately
    if isinstance(raw_text, str):
        return raw_text

    # Handle list/tuple - join into a single string
    if isinstance(raw_text, (list, tuple)):
        if len(raw_text) == 0:
            return ""
        flat: List[str] = []
        for t in raw_text:
            if t is None:
                continue
            if isinstance(t, (list, tuple)):
                flat.extend(str(x) for x in t if x is not None)
            else:
                flat.append(str(t))
        result = " ".join(flat) if flat else ""
        # CRITICAL: Ensure we return a string, not a list
        return str(result) if not isinstance(result, str) else result

    # Handle dict - try to extract text-like fields
    if isinstance(raw_text, dict):
        # Try common text field names
        for key in ["text", "content", "sentence", "input"]:
            if key in raw_text and isinstance(raw_text[key], str):
                return raw_text[key]
        # Fall back to JSON serialization
        try:
            result = json.dumps(raw_text, ensure_ascii=False)
            return str(result) if not isinstance(result, str) else result
        except Exception:
            result = str(raw_text)
            return str(result) if not isinstance(result, str) else result

    # Handle other types - convert to string
    try:
        result = str(raw_text)
        # Ensure it's not "None" string
        if result.lower() == "none":
            return ""
        # CRITICAL: Double-check it's actually a string
        return str(result) if not isinstance(result, str) else result
    except Exception:
        return ""


def encode_annotations_to_labels(
    text: str,
    annotations: List[List[Any]],
    offsets: List[Tuple[int, int]],
    label2id: Dict[str, int],
) -> List[int]:
    """
    Encode character-level annotations to token-level labels.

    Args:
        text: Original text string.
        annotations: List of annotations as [start, end, entity_type].
        offsets: List of token offset tuples (start, end).
        label2id: Mapping from label strings to integer IDs.

    Returns:
        List of label IDs corresponding to each token.
    """
    labels = []
    for start, end in offsets:
        lab = "O"
        for ann_start, ann_end, ent in annotations:
            if not (end <= ann_start or start >= ann_end):
                lab = ent
                break
        labels.append(label2id.get(lab, label2id["O"]))
    return labels


# Import Dataset lazily - only when ResumeNERDataset is actually instantiated
# This allows build_label_list to be imported without requiring torch
def _get_dataset_class():
    """Lazy import of Dataset class."""
    from torch.utils.data import Dataset
    return Dataset

class ResumeNERDataset:
    """Dataset for Resume NER token classification.
    
    This class behaves like torch.utils.data.Dataset but inherits dynamically
    to avoid requiring torch at module import time.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        label2id: Dict[str, int],
    ):
        # Validate samples are dictionaries
        validated_samples = []
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(
                    f"Sample at index {i} is not a dictionary: {type(sample)} - {sample}"
                )
            validated_samples.append(sample)

        self.samples = validated_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Encode a single sample.

        For fast tokenizers we request ``return_offsets_mapping`` so that we can
        align character-level annotations to token labels.  Some slow (Python)
        tokenizers – including the DeBERTa v2/v3 slow tokenizers – do **not**
        support ``return_offsets_mapping`` and will raise ``NotImplementedError``.
        In that case we fall back to a simpler behaviour where all tokens are
        labelled as ``O`` so that the training loop can still run and the
        orchestration pipeline can be validated.
        """
        item = self.samples[idx]

        # Ensure item is a dictionary
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected dict for sample at index {idx}, got {type(item)}: {item}"
            )

        # Get and normalize text
        raw_text = item.get("text", "")
        text = normalize_text_for_tokenization(raw_text)

        # Ensure text is a string after normalization
        if not isinstance(text, str):
            raise TypeError(
                f"Text must be a string after normalization, got {type(text)}: {text}. "
                f"Original raw_text type: {type(raw_text)}, value: {raw_text}. "
                f"Item keys: {list(item.keys()) if isinstance(item, dict) else 'N/A'}"
            )

        # Ensure text is a proper string (not bytes or other string-like types)
        # Convert to plain str if it's a subclass
        if not type(text) is str:  # Use 'is' not 'isinstance' to catch subclasses
            text = str(text)

        # Ensure text is not empty (tokenizer may fail on empty strings)
        if not text or not text.strip():
            # Use a placeholder if text is empty
            text = " "

        # Final validation: ensure text is a plain Python str
        if not isinstance(text, str) or not type(text) is str:
            raise TypeError(
                f"Text validation failed: type={type(text)}, isinstance(str)={isinstance(text, str)}, "
                f"value={repr(text[:100])}"
            )

        annotations = item.get("annotations", []) or []

        # Ensure annotations is a list
        if not isinstance(annotations, list):
            annotations = []

        supports_offsets = bool(getattr(self.tokenizer, "is_fast", False))

        # Final validation: ensure text is exactly a plain Python str (not a subclass)
        if type(text) is not str:
            text = str(text)

        # Handle case where normalize_text_for_tokenization might return a list/tuple
        # The tokenizer expects a single string, not a sequence
        if isinstance(text, (list, tuple)):
            # If text is a list/tuple, join it into a string
            text = " ".join(str(t) for t in text if t is not None)
        elif not isinstance(text, str):
            # Force conversion to string for any other type
            text = str(text)

        # Additional validation: ensure text is valid UTF-8
        # Handle surrogate characters and other invalid UTF-8 by cleaning them
        try:
            # Try to encode/decode to check validity
            text.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # Clean invalid UTF-8 characters (surrogates, etc.)
            # Replace invalid characters with a placeholder or remove them
            try:
                # Try encoding with error handling
                text = text.encode(
                    'utf-8', errors='replace').decode('utf-8', errors='replace')
            except Exception:
                # If that fails, remove surrogate characters manually
                # Surrogate characters are in the range \ud800-\udfff
                text = re.sub(r'[\ud800-\udfff]', '', text)
                # If text becomes empty after cleaning, use placeholder
                if not text.strip():
                    text = " "

        # Final type check - must be a plain Python string
        if not isinstance(text, str) or type(text) is not str:
            raise TypeError(
                f"CRITICAL: Text is not a string before tokenizer call. "
                f"Type: {type(text)}, Value: {repr(text[:100])}, "
                f"Index: {idx}, Item keys: {list(item.keys())}, "
                f"Raw text type: {type(raw_text)}, Raw text: {repr(raw_text[:100]) if isinstance(raw_text, str) else raw_text}"
            )

        # Final check before tokenizer call
        try:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=supports_offsets,
                return_tensors="pt",
            )
        except TypeError as e:
            # Provide detailed error information
            error_msg = (
                f"Tokenizer TypeError at dataset index {idx}. "
                f"Text type: {type(text)}, type is str: {type(text) is str}, "
                f"isinstance(str): {isinstance(text, str)}, "
                f"Text repr (first 200 chars): {repr(text[:200])}, "
                f"Text length: {len(text)}, "
                f"Item keys: {list(item.keys())}, "
                f"Item type: {type(item)}, "
                f"Raw text type: {type(raw_text)}, Raw text value: {repr(raw_text[:100]) if isinstance(raw_text, str) else raw_text}, "
                f"Original error: {str(e)}"
            )
            raise TypeError(error_msg) from e

        if supports_offsets and "offset_mapping" in encoded:
            offsets = encoded.pop("offset_mapping")[0].tolist()
            labels = encode_annotations_to_labels(
                text, annotations, offsets, self.label2id
            )
        else:
            seq_len = encoded["input_ids"].shape[1]
            labels = [self.label2id.get("O", 0)] * seq_len

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        import torch
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

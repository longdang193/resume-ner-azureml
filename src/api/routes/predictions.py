"""Prediction endpoints for NER API."""

import time
import logging
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from fastapi import HTTPException, status, UploadFile, File, Form

from ..config import APIConfig
from ..model_loader import get_engine, is_model_loaded
from ..models import (
    TextRequest,
    BatchTextRequest,
    PredictionResponse,
    BatchPredictionResponse,
)
from ..exceptions import (
    InferenceError,
    TextExtractionError,
    InvalidFileTypeError,
    FileSizeExceededError,
)
from ..extractors import (
    extract_text_from_pdf,
    extract_text_from_image,
    detect_file_type,
    validate_file,
)
from ..response_converters import convert_entities_to_response

logger = logging.getLogger(__name__)


async def predict_debug(request: TextRequest):
    """Debug endpoint to inspect predictions and entity extraction."""
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        engine = get_engine()

        # Get raw predictions
        logits, tokens, tokenizer_output, offset_mapping = engine.predict_tokens(
            request.text)

        # Handle logits shape - ensure it's 2D (seq_len, num_labels)
        if len(logits.shape) == 1:
            logits = logits.reshape(1, -1)
        predictions = np.argmax(logits, axis=-1)
        # Flatten predictions if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        labels = [engine.id2label.get(int(p), "O") for p in predictions]

        # Get attention mask
        attention_mask = tokenizer_output.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask[0].tolist()
        else:
            attention_mask = [1] * len(tokens)

        # Process offset mapping (simplified - just for debug display)
        token_offsets = []
        if offset_mapping is not None:
            try:
                if hasattr(offset_mapping, 'shape') and len(offset_mapping.shape) == 2:
                    for i in range(min(len(offset_mapping), len(tokens))):
                        offset = offset_mapping[i]
                        if hasattr(offset, '__len__') and len(offset) >= 2:
                            start, end = int(offset[0]), int(offset[1])
                        else:
                            start, end = int(offset), int(offset)
                        if start == 0 and end == 0:
                            token_offsets.append(None)
                        else:
                            token_offsets.append((start, end))
                else:
                    token_offsets = [None] * len(tokens)
            except Exception:
                token_offsets = [None] * len(tokens)
        else:
            token_offsets = [None] * len(tokens)

        # Get entities
        entities = engine.decode_entities(
            request.text,
            logits,
            tokens,
            tokenizer_output,
            offset_mapping,
            return_confidence=True,
        )

        # Build debug response
        debug_info = {
            "text": request.text,
            "text_length": len(request.text),
            "num_tokens": len(tokens),
            "num_labels": len(labels),
            "non_o_labels_count": sum(1 for l in labels if l != "O"),
            "offset_mapping_available": offset_mapping is not None,
            "offset_mapping_shape": list(offset_mapping.shape) if offset_mapping is not None else None,
            "num_entities_extracted": len(entities),
            "sample_tokens": tokens[:20],
            "sample_labels": labels[:20],
            "sample_offsets": token_offsets[:20],
            "sample_attention_mask": attention_mask[:20],
            "id2label_keys": list(engine.id2label.keys())[:20],
            "entities": [
                {
                    "text": e["text"],
                    "label": e["label"],
                    "start": e["start"],
                    "end": e["end"],
                    "confidence": e.get("confidence"),
                }
                for e in entities
            ],
        }

        return debug_info

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug endpoint error: {str(e)}",
        )


async def predict(request: TextRequest):
    """Single text prediction endpoint."""
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        engine = get_engine()
        start_time = time.time()

        # Direct prediction (removed redundant debug code that computed predictions twice)
        entities_dict = engine.predict(request.text, return_confidence=True)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Convert to Entity models using shared converter
        entities = convert_entities_to_response(entities_dict)

        return PredictionResponse(
            entities=entities,
            processing_time_ms=processing_time,
        )
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


def _process_single_text(text: str, index: int, engine, max_text_time: float) -> tuple:
    """Process a single text and return result with index for ordering."""
    text_start = time.time()
    try:
        # Validate text before processing
        if not isinstance(text, str):
            raise ValueError(f"Text {index+1} is not a string: {type(text)}")
        if not text.strip():
            raise ValueError(f"Text {index+1} is empty")
        if len(text) > APIConfig.MAX_SEQUENCE_LENGTH * 10:  # Rough estimate
            raise ValueError(
                f"Text {index+1} too long: {len(text)} characters (max ~{APIConfig.MAX_SEQUENCE_LENGTH * 10})")

        logger.debug(f"Processing text {index+1} ({len(text)} chars)")
        entities_dict = engine.predict(text, return_confidence=True)
        text_time = (time.time() - text_start) * 1000

        # Check if processing took too long
        if (time.time() - text_start) > max_text_time:
            raise TimeoutError(f"Text {index+1} processing exceeded {max_text_time}s")

        entities = convert_entities_to_response(entities_dict)
        return (index, PredictionResponse(entities=entities, processing_time_ms=text_time), None)
    except Exception as e:
        elapsed = (time.time() - text_start) * 1000
        error_msg = f"Text {index+1} failed: {str(e)} (took {elapsed:.1f}ms)"
        logger.error(error_msg, exc_info=True)
        return (index, PredictionResponse(entities=[], processing_time_ms=elapsed), error_msg)


async def predict_batch(request: BatchTextRequest):
    """Batch text prediction endpoint with parallel processing."""
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    if len(request.texts) > APIConfig.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum of {APIConfig.MAX_BATCH_SIZE}",
        )

    try:
        engine = get_engine()
        start_time = time.time()
        max_text_time = 30.0  # Maximum time per text in seconds

        # Use parallel processing for batch requests (I/O-bound inference can benefit from threading)
        # For small batches, sequential may be faster due to overhead, but parallel helps with larger batches
        use_parallel = len(request.texts) >= 3  # Use parallel for 3+ items
        
        if use_parallel:
            # Process texts in parallel using ThreadPoolExecutor
            # Note: ONNX inference is CPU-bound, but GIL allows some parallelism for I/O
            max_workers = min(4, len(request.texts))  # Limit to 4 workers to avoid contention
            results = [None] * len(request.texts)
            errors = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_process_single_text, text, i, engine, max_text_time): i
                    for i, text in enumerate(request.texts)
                }
                
                for future in as_completed(futures):
                    try:
                        index, prediction, error = future.result()
                        results[index] = prediction
                        if error:
                            errors.append(error)
                    except Exception as e:
                        idx = futures[future]
                        logger.error(f"Text {idx+1} processing failed: {e}", exc_info=True)
                        errors.append(f"Text {idx+1} failed: {str(e)}")
                        results[idx] = PredictionResponse(entities=[], processing_time_ms=0)
            
            predictions = results
        else:
            # Sequential processing for small batches (lower overhead)
            predictions = []
            errors = []
            for i, text in enumerate(request.texts):
                _, prediction, error = _process_single_text(text, i, engine, max_text_time)
                predictions.append(prediction)
                if error:
                    errors.append(error)

        total_time = (time.time() - start_time) * 1000

        # If all texts failed, raise an error
        if len(errors) == len(request.texts):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"All texts failed: {'; '.join(errors)}",
            )

        # If some texts failed, log warning but return partial results
        if errors:
            logger.warning(
                f"Batch processing completed with {len(errors)} errors: {'; '.join(errors)}")

        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time,
        )
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


async def predict_file(
    file: UploadFile = File(...),
    extractor: Optional[str] = Form(None),
):
    """File upload prediction endpoint (PDF or image)."""
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Validate and read file
        file_content = await validate_file(file)

        # Detect file type
        file_type = detect_file_type(file_content, file.filename or "")

        # Extract text
        start_extract = time.time()
        if file_type == "application/pdf":
            pdf_extractor = _normalize_extractor(extractor, APIConfig.PDF_EXTRACTOR)
            extracted_text = extract_text_from_pdf(file_content, pdf_extractor)
        elif file_type.startswith("image/"):
            ocr_extractor = _normalize_extractor(extractor, APIConfig.OCR_EXTRACTOR)
            extracted_text = extract_text_from_image(file_content, ocr_extractor)
        else:
            raise InvalidFileTypeError(f"Unsupported file type: {file_type}")

        extract_time = (time.time() - start_extract) * 1000

        # Run NER prediction
        engine = get_engine()
        start_infer = time.time()
        entities_dict = engine.predict(extracted_text, return_confidence=True)
        infer_time = (time.time() - start_infer) * 1000

        # Convert to Entity models
        entities = convert_entities_to_response(entities_dict)

        total_time = extract_time + infer_time

        return PredictionResponse(
            entities=entities,
            processing_time_ms=total_time,
            extracted_text=extracted_text,
        )
    except (TextExtractionError, InvalidFileTypeError, FileSizeExceededError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


def _normalize_extractor(extractor: Optional[str], default: str) -> str:
    """Normalize extractor parameter, handling Swagger UI placeholder 'string'."""
    if not extractor or not extractor.strip() or extractor.lower() == "string":
        return default
    return extractor.strip()


async def _process_single_file(file: UploadFile, index: int, extractor: Optional[str], engine) -> tuple:
    """Process a single file and return result with index for ordering."""
    file_start = time.time()
    try:
        # Validate and read file
        file_content = await validate_file(file)

        # Detect file type
        file_type = detect_file_type(file_content, file.filename or "")

        # Extract text
        if file_type == "application/pdf":
            pdf_extractor = _normalize_extractor(extractor, APIConfig.PDF_EXTRACTOR)
            extracted_text = extract_text_from_pdf(file_content, pdf_extractor)
        elif file_type.startswith("image/"):
            ocr_extractor = _normalize_extractor(extractor, APIConfig.OCR_EXTRACTOR)
            extracted_text = extract_text_from_image(file_content, ocr_extractor)
        else:
            raise InvalidFileTypeError(f"Unsupported file type: {file_type}")

        # Run NER prediction
        entities_dict = engine.predict(extracted_text, return_confidence=True)

        file_time = (time.time() - file_start) * 1000

        # Convert to Entity models
        entities = convert_entities_to_response(entities_dict)

        return (index, PredictionResponse(
            entities=entities,
            processing_time_ms=file_time,
            extracted_text=extracted_text,
        ), None)
    except Exception as e:
        elapsed = (time.time() - file_start) * 1000
        error_msg = f"File {index+1} ({file.filename or 'unknown'}) failed: {str(e)} (took {elapsed:.1f}ms)"
        logger.error(error_msg, exc_info=True)
        return (index, PredictionResponse(
            entities=[],
            processing_time_ms=elapsed,
            extracted_text="",
        ), error_msg)


async def predict_file_batch(
    files: List[UploadFile] = File(...),
    extractor: Optional[str] = Form(None),
):
    """Batch file upload prediction endpoint with parallel processing."""
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    if len(files) > APIConfig.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum of {APIConfig.MAX_BATCH_SIZE}",
        )

    try:
        engine = get_engine()
        start_time = time.time()

        # Use parallel processing for batch file requests (I/O-bound file reading + extraction)
        # For small batches, sequential may be faster due to overhead
        use_parallel = len(files) >= 2  # Use parallel for 2+ files
        
        if use_parallel:
            # Process files in parallel using asyncio.gather (better for async I/O)
            tasks = [
                _process_single_file(file, i, extractor, engine)
                for i, file in enumerate(files)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            predictions = []
            errors = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"File {i+1} processing failed: {result}", exc_info=True)
                    errors.append(f"File {i+1} failed: {str(result)}")
                    predictions.append(PredictionResponse(entities=[], processing_time_ms=0, extracted_text=""))
                else:
                    index, prediction, error = result
                    predictions.append(prediction)
                    if error:
                        errors.append(error)
        else:
            # Sequential processing for small batches
            predictions = []
            errors = []
            for i, file in enumerate(files):
                _, prediction, error = await _process_single_file(file, i, extractor, engine)
                predictions.append(prediction)
                if error:
                    errors.append(error)

        total_time = (time.time() - start_time) * 1000

        # If all files failed, raise an error
        if len(errors) == len(files):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"All files failed: {'; '.join(errors)}",
            )

        # If some files failed, log warning but return partial results
        if errors:
            logger.warning(
                f"Batch file processing completed with {len(errors)} errors: {'; '.join(errors)}")

        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time,
        )
    except (TextExtractionError, InvalidFileTypeError, FileSizeExceededError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

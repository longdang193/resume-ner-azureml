"""Inference performance benchmarking script.

Measures actual inference latency and throughput for NER models,
replacing parameter-count proxies with real performance data.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    device: Optional[str] = None,
) -> tuple:
    """
    Load model and tokenizer from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory containing model files.
        device: Device to load model on ('cuda', 'cpu', or None for auto-detect).

    Returns:
        Tuple of (model, tokenizer, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
    model.to(device_obj)
    model.eval()
    
    return model, tokenizer, device_obj


def benchmark_single_inference(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
) -> float:
    """
    Measure single document inference time.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        text: Input text to process.
        device: Device to run inference on.
        max_length: Maximum sequence length.

    Returns:
        Inference time in milliseconds.
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Measure inference time
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    return latency_ms


def benchmark_batch_inference(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Measure batch inference performance.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        texts: List of input texts to process.
        device: Device to run inference on.
        max_length: Maximum sequence length.
        num_iterations: Number of iterations to measure.
        warmup_iterations: Number of warmup iterations before measurement.

    Returns:
        Dictionary with latency statistics (mean, median, p95, p99) and throughput.
    """
    latencies = []
    
    # Warmup
    for _ in range(warmup_iterations):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
    
    # Actual measurement
    for _ in range(num_iterations):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
    
    # Calculate statistics
    if not latencies:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "throughput_docs_per_sec": 0.0,
        }
    
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    
    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    p95_idx = int(len(sorted_latencies) * 0.95)
    p99_idx = int(len(sorted_latencies) * 0.99)
    p95_ms = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
    p99_ms = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
    
    # Calculate throughput (documents per second)
    batch_size = len(texts)
    throughput_docs_per_sec = batch_size / (mean_ms / 1000)
    
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "throughput_docs_per_sec": throughput_docs_per_sec,
    }


def benchmark_model(
    checkpoint_dir: Path,
    test_texts: List[str],
    batch_sizes: List[int] = [1, 8, 16],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: Optional[str] = None,
    max_length: int = 512,
) -> Dict:
    """
    Benchmark model inference performance across different batch sizes.

    Args:
        checkpoint_dir: Path to checkpoint directory.
        test_texts: List of test texts to use for benchmarking.
        batch_sizes: List of batch sizes to test.
        num_iterations: Number of iterations per batch size.
        warmup_iterations: Number of warmup iterations.
        device: Device to use ('cuda', 'cpu', or None for auto-detect).
        max_length: Maximum sequence length.

    Returns:
        Dictionary with benchmark results for each batch size.
    """
    print(f"Loading model from {checkpoint_dir}...")
    model, tokenizer, device_obj = load_model_from_checkpoint(checkpoint_dir, device)
    print(f"Model loaded on device: {device_obj}")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size {batch_size}...")
        
        # Create batch
        batch_texts = test_texts[:batch_size]
        if len(batch_texts) < batch_size:
            # Repeat texts if needed
            batch_texts = (batch_texts * ((batch_size // len(batch_texts)) + 1))[:batch_size]
        
        # Benchmark this batch size
        batch_results = benchmark_batch_inference(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            device=device_obj,
            max_length=max_length,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
        )
        
        results[f"batch_{batch_size}"] = batch_results
        
        print(f"  Mean latency: {batch_results['mean_ms']:.2f} ms")
        print(f"  P95 latency: {batch_results['p95_ms']:.2f} ms")
        print(f"  Throughput: {batch_results['throughput_docs_per_sec']:.2f} docs/sec")
    
    # Add metadata
    results["device"] = str(device_obj)
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    return results


def main():
    """CLI entry point for benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Benchmark NER model inference performance"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data JSON file (list of texts or list of dicts with 'text' field)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16],
        help="Batch sizes to test (default: 1 8 16)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per batch size (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file for benchmark results",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    
    args = parser.parse_args()
    
    # Load test data
    checkpoint_dir = Path(args.checkpoint)
    test_data_path = Path(args.test_data)
    output_path = Path(args.output)
    
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    if not test_data_path.exists():
        raise ValueError(f"Test data file not found: {test_data_path}")
    
    # Load test texts
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Extract texts from test data
    if isinstance(test_data, list):
        if len(test_data) > 0 and isinstance(test_data[0], dict):
            # List of dicts with 'text' field
            test_texts = [item.get("text", "") for item in test_data if item.get("text")]
        else:
            # List of strings
            test_texts = [str(item) for item in test_data if item]
    else:
        raise ValueError("Test data must be a list of texts or list of dicts with 'text' field")
    
    if not test_texts:
        raise ValueError("No test texts found in test data file")
    
    print(f"Loaded {len(test_texts)} test texts")
    
    # Run benchmark
    results = benchmark_model(
        checkpoint_dir=checkpoint_dir,
        test_texts=test_texts,
        batch_sizes=args.batch_sizes,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        device=args.device,
        max_length=args.max_length,
    )
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {output_path}")


if __name__ == "__main__":
    main()




















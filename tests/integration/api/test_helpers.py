"""Test helpers for FastAPI local server integration tests.

This module provides utilities for server lifecycle management, HTTP client,
and performance validation.
"""

import os
import socket
import subprocess
import sys
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import httpx
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ServerHandle:
    """Handle for a running server process."""
    process: subprocess.Popen
    base_url: str
    host: str
    port: int


@dataclass
class Response:
    """HTTP response wrapper."""
    status_code: int
    json_data: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ValidationResult:
    """Performance validation result."""
    passed: bool
    endpoint: str
    metric: str
    measured: float
    threshold: float
    message: str


class ServerManager:
    """Manages FastAPI server process lifecycle."""

    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize server manager.

        Args:
            root_dir: Project root directory (default: auto-detect)
        """
        if root_dir is None:
            # Auto-detect root directory (4 levels up from this file)
            root_dir = Path(__file__).parent.parent.parent.parent
        self.root_dir = Path(root_dir)
        self.python_executable = Path(
            sys.executable) if 'sys' in globals() else Path("python")

    def check_port_in_use(self, host: str, port: int) -> bool:
        """
        Check if a port is already in use.

        Args:
            host: Host address
            port: Port number

        Returns:
            True if port is in use, False otherwise
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.bind((host, port))
            sock.close()
            return False
        except OSError:
            return True
        except Exception:
            return True
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def find_available_port(self, host: str, start_port: int, end_port: int) -> Optional[int]:
        """
        Find an available port in the given range.

        Args:
            host: Host address
            start_port: Start of port range
            end_port: End of port range

        Returns:
            Available port number or None if none found
        """
        for port in range(start_port, end_port + 1):
            if not self.check_port_in_use(host, port):
                return port
        return None

    def is_server_ready(self, base_url: str, timeout: int = 2) -> bool:
        """
        Check if server is responding to health checks.

        Args:
            base_url: Server base URL
            timeout: Timeout in seconds

        Returns:
            True if server is ready, False otherwise
        """
        try:
            response = httpx.get(f"{base_url}/health", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def start_server(
        self,
        onnx_path: Path,
        checkpoint_dir: Path,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout: int = 45,
        log_level: str = "INFO"
    ) -> ServerHandle:
        """
        Start server and wait for readiness.

        Args:
            onnx_path: Path to ONNX model file
            checkpoint_dir: Path to checkpoint directory
            host: Server host address
            port: Server port number
            timeout: Maximum seconds to wait for server readiness
            log_level: Logging level

        Returns:
            ServerHandle with process and base URL

        Raises:
            RuntimeError: If server fails to start or become ready
        """
        # Check if port is available
        if self.check_port_in_use(host, port):
            raise RuntimeError(f"Port {port} is already in use on {host}")

        # Build command
        cmd = [
            str(self.python_executable),
            "-m", "src.api.cli.run_api",
            "--onnx-model", str(onnx_path),
            "--checkpoint", str(checkpoint_dir),
            "--host", host,
            "--port", str(port),
            "--log-level", log_level,
        ]

        # Set environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.root_dir)

        # Start process - redirect stderr to DEVNULL to avoid blocking issues
        # We'll only read stderr if the process fails
        logger.info(f"Starting server: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # Keep PIPE for error reading, but don't read continuously
            cwd=str(self.root_dir),
            env=env,
            text=True,
            bufsize=1,  # Line buffered
        )

        base_url = f"http://{host}:{port}"

        # Wait for server to be ready
        start_time = time.time()

        try:
            while (time.time() - start_time) < timeout:
                # Check if process has exited
                return_code = process.poll()
                if return_code is not None:
                    # Process exited - read stderr for error message
                    stderr_content = ""
                    try:
                        # Read available stderr (non-blocking)
                        import select
                        if sys.platform != "win32":
                            # Unix: use select to check if data is available
                            if select.select([process.stderr], [], [], 0.1)[0]:
                                stderr_content = process.stderr.read()
                        else:
                            # Windows: try to read what's available
                            # Set stderr to non-blocking mode is tricky on Windows
                            # Just try to read - it might block briefly but process is dead
                            try:
                                stderr_content = process.stderr.read()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    error_msg = f"Server process exited with code {return_code}"
                    if stderr_content:
                        stderr_lines = stderr_content.strip().split('\n')
                        # Check for common dependency errors
                        stderr_text = stderr_content.lower()
                        if "python-multipart" in stderr_text:
                            error_msg += "\n\n" + "="*60
                            error_msg += "\nMISSING DEPENDENCY: python-multipart"
                            error_msg += "\n" + "="*60
                            error_msg += "\nThe FastAPI server requires 'python-multipart' for file upload endpoints."
                            error_msg += "\nInstall it with: pip install python-multipart"
                            error_msg += "\n" + "="*60
                        # Get last 30 lines
                        error_msg += f"\n\nStderr (last 30 lines):\n" + \
                            "\n".join(stderr_lines[-30:])
                    raise RuntimeError(error_msg)

                # Check if server is ready
                if self.is_server_ready(base_url, timeout=2):
                    logger.info(f"Server is ready at {base_url}")
                    # Server is running - close stderr to free resources
                    # Don't read it anymore since server is working
                    try:
                        process.stderr.close()
                    except Exception:
                        pass
                    return ServerHandle(process=process, base_url=base_url, host=host, port=port)

                # Log progress every 5 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    logger.debug(
                        f"Waiting for server... ({elapsed:.1f}s elapsed, process running: {process.poll() is None})")

                time.sleep(0.5)

            # Timeout - try to read stderr for diagnostics
            stderr_content = ""
            try:
                # Try to read stderr (non-blocking)
                if sys.platform != "win32":
                    import select
                    if select.select([process.stderr], [], [], 0.1)[0]:
                        stderr_content = process.stderr.read()
                else:
                    try:
                        stderr_content = process.stderr.read()
                    except Exception:
                        pass
            except Exception:
                pass

            # Kill the process
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                    process.wait()
                except Exception:
                    pass

            error_msg = f"Server failed to become ready within {timeout} seconds"
            if stderr_content:
                stderr_lines = stderr_content.strip().split('\n')
                stderr_text = stderr_content.lower()
                if "python-multipart" in stderr_text:
                    error_msg += "\n\n" + "="*60
                    error_msg += "\nMISSING DEPENDENCY: python-multipart"
                    error_msg += "\n" + "="*60
                    error_msg += "\nThe FastAPI server requires 'python-multipart' for file upload endpoints."
                    error_msg += "\nInstall it with: pip install python-multipart"
                    error_msg += "\n" + "="*60
                error_msg += f"\n\nStderr (last 30 lines):\n" + \
                    "\n".join(stderr_lines[-30:])
            error_msg += f"\nProcess status: {'running' if process.poll() is None else f'exited with code {process.returncode}'}"
            raise RuntimeError(error_msg)
        finally:
            # Clean up stderr if still open
            try:
                if process.stderr and not process.stderr.closed:
                    process.stderr.close()
            except Exception:
                pass

    def stop_server(self, handle: ServerHandle, timeout: int = 5) -> bool:
        """
        Stop server gracefully.

        Args:
            handle: ServerHandle from start_server
            timeout: Timeout for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        if handle.process.poll() is not None:
            # Already stopped
            return True

        try:
            handle.process.terminate()
            handle.process.wait(timeout=timeout)
            logger.info(f"Server stopped gracefully")
            return True
        except subprocess.TimeoutExpired:
            logger.warning(f"Server did not stop gracefully, killing process")
            handle.process.kill()
            handle.process.wait()
            return False


class APIClient:
    """HTTP client for making requests to the FastAPI server."""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize API client.

        Args:
            base_url: Server base URL (e.g., "http://127.0.0.1:8000")
            timeout: Default request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Response:
        """
        Make HTTP request and measure latency.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: Endpoint path (e.g., "/health")
            **kwargs: Additional arguments for httpx request

        Returns:
            Response object with status, JSON, and latency
        """
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            if method.upper() == "GET":
                http_response = self.client.get(url, **kwargs)
            elif method.upper() == "POST":
                http_response = self.client.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")

            latency_ms = (time.time() - start_time) * 1000

            # Parse JSON if possible
            json_data = None
            try:
                json_data = http_response.json()
            except Exception:
                pass

            return Response(
                status_code=http_response.status_code,
                json_data=json_data,
                text=http_response.text,
                latency_ms=latency_ms
            )
        except httpx.TimeoutException as e:
            latency_ms = (time.time() - start_time) * 1000
            raise RuntimeError(
                f"Request to {url} timed out after {latency_ms:.1f}ms") from e
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise RuntimeError(f"Request to {url} failed: {e}") from e

    def health_check(self) -> Response:
        """GET /health"""
        return self._make_request("GET", "/health")

    def model_info(self) -> Response:
        """GET /info"""
        return self._make_request("GET", "/info")

    def predict(self, text: str) -> Response:
        """POST /predict"""
        return self._make_request("POST", "/predict", json={"text": text})

    def predict_batch(self, texts: List[str]) -> Response:
        """POST /predict/batch"""
        return self._make_request("POST", "/predict/batch", json={"texts": texts})

    def predict_file(
        self,
        file_path: Path,
        extractor: Optional[str] = None
    ) -> Response:
        """
        POST /predict/file

        Args:
            file_path: Path to file to upload
            extractor: Optional extractor name
        """
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {}
            if extractor:
                data["extractor"] = extractor
            return self._make_request("POST", "/predict/file", files=files, data=data)

    def predict_file_batch(
        self,
        file_paths: List[Path],
        extractor: Optional[str] = None
    ) -> Response:
        """
        POST /predict/file/batch

        Args:
            file_paths: List of file paths to upload
            extractor: Optional extractor name
        """
        # Open all files first
        file_handles = []
        files = []
        try:
            for file_path in file_paths:
                file_handle = open(file_path, "rb")
                file_handles.append(file_handle)
                # FastAPI expects files parameter name
                files.append(
                    ("files", (file_path.name, file_handle, "application/octet-stream")))

            data = {}
            if extractor:
                data["extractor"] = extractor

            return self._make_request("POST", "/predict/file/batch", files=files, data=data)
        finally:
            # Close all file handles
            for file_handle in file_handles:
                try:
                    file_handle.close()
                except Exception:
                    pass

    def predict_debug(self, text: str) -> Response:
        """POST /predict/debug"""
        return self._make_request("POST", "/predict/debug", json={"text": text})

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def load_test_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load test configuration from YAML file.

    Args:
        config_path: Path to config file (default: config/test/api_local_server.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        root_dir = Path(__file__).parent.parent.parent.parent
        config_path = root_dir / "config" / "test" / "api_local_server.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_latency(
    endpoint: str,
    measured_latency_ms: float,
    thresholds: Dict[str, float],
    percentile: Optional[str] = None
) -> ValidationResult:
    """
    Validate latency against thresholds.

    Args:
        endpoint: Endpoint name (e.g., "predict")
        measured_latency_ms: Measured latency in milliseconds
        thresholds: Dictionary with keys like "p50_ms", "p95_ms", "max_ms"
        percentile: Optional percentile being validated (e.g., "p50", "p95")

    Returns:
        ValidationResult with pass/fail status
    """
    if percentile:
        threshold_key = f"{percentile}_ms"
    else:
        # Use max threshold by default
        threshold_key = "max_ms"

    if threshold_key not in thresholds:
        # Try to find any threshold
        if "max_ms" in thresholds:
            threshold_key = "max_ms"
        elif "p95_ms" in thresholds:
            threshold_key = "p95_ms"
        elif "p50_ms" in thresholds:
            threshold_key = "p50_ms"
        else:
            return ValidationResult(
                passed=False,
                endpoint=endpoint,
                metric="latency",
                measured=measured_latency_ms,
                threshold=0.0,
                message=f"No threshold found for {endpoint}"
            )

    threshold = thresholds[threshold_key]
    passed = measured_latency_ms <= threshold

    message = (
        f"{endpoint} latency: {measured_latency_ms:.1f}ms "
        f"(threshold: {threshold}ms for {threshold_key})"
    )

    return ValidationResult(
        passed=passed,
        endpoint=endpoint,
        metric="latency",
        measured=measured_latency_ms,
        threshold=threshold,
        message=message
    )

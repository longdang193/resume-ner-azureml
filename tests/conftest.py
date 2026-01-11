"""Global pytest configuration and fixtures.

This file is automatically discovered by pytest and applies to all tests.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add project root and src to Python path for all tests
# IMPORTANT: Add SRC_DIR first to ensure src modules are found before test modules
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"

# Add SRC_DIR first to avoid namespace collisions with test directories
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))  # Insert after SRC_DIR

# Install compatibility shim import finders early to handle submodule imports
# This ensures that imports like "from selection.selection_logic import ..." work
# even when selection module hasn't been imported yet
try:
    import importlib
    import importlib.util
    from importlib.abc import MetaPathFinder, Loader
    from importlib.util import spec_from_loader
    
    class ProxyLoader(Loader):
        """Loader that loads from evaluation.* or deployment.* but registers as old module names."""
        def __init__(self, target_module_name):
            self.target_module_name = target_module_name
        
        def create_module(self, spec):
            # Import the actual module and register it under the proxy name
            actual_module = importlib.import_module(self.target_module_name)
            # Register it under the proxy name in sys.modules
            sys.modules[spec.name] = actual_module
            return actual_module
        
        def exec_module(self, module):
            # Module is already loaded, nothing to do
            pass
    
    class SelectionSubmoduleFinder(MetaPathFinder):
        """Custom finder for selection.* submodules."""
        def find_spec(self, name, path, target=None):
            if name.startswith('selection.') and name != 'selection':
                submodule_name = name.replace('selection.', 'evaluation.selection.', 1)
                try:
                    spec = importlib.util.find_spec(submodule_name)
                    if spec is not None:
                        loader = ProxyLoader(submodule_name)
                        return spec_from_loader(name, loader)
                except (ImportError, ValueError):
                    pass
            return None
    
    class BenchmarkingSubmoduleFinder(MetaPathFinder):
        """Custom finder for benchmarking.* submodules."""
        def find_spec(self, name, path, target=None):
            if name.startswith('benchmarking.') and name != 'benchmarking':
                submodule_name = name.replace('benchmarking.', 'evaluation.benchmarking.', 1)
                try:
                    spec = importlib.util.find_spec(submodule_name)
                    if spec is not None:
                        loader = ProxyLoader(submodule_name)
                        return spec_from_loader(name, loader)
                except (ImportError, ValueError):
                    pass
            return None
    
    class ConversionSubmoduleFinder(MetaPathFinder):
        """Custom finder for conversion.* submodules."""
        def find_spec(self, name, path, target=None):
            if name.startswith('conversion.') and name != 'conversion':
                submodule_name = name.replace('conversion.', 'deployment.conversion.', 1)
                try:
                    spec = importlib.util.find_spec(submodule_name)
                    if spec is not None:
                        loader = ProxyLoader(submodule_name)
                        return spec_from_loader(name, loader)
                except (ImportError, ValueError):
                    pass
            return None
    
    class ApiSubmoduleFinder(MetaPathFinder):
        """Custom finder for api.* submodules."""
        def find_spec(self, name, path, target=None):
            if name.startswith('api.') and name != 'api':
                submodule_name = name.replace('api.', 'deployment.api.', 1)
                try:
                    spec = importlib.util.find_spec(submodule_name)
                    if spec is not None:
                        loader = ProxyLoader(submodule_name)
                        return spec_from_loader(name, loader)
                except (ImportError, ValueError):
                    pass
            return None
    
    # Install finders if not already installed
    _selection_finder_installed = any(isinstance(f, SelectionSubmoduleFinder) for f in sys.meta_path)
    _benchmarking_finder_installed = any(isinstance(f, BenchmarkingSubmoduleFinder) for f in sys.meta_path)
    _conversion_finder_installed = any(isinstance(f, ConversionSubmoduleFinder) for f in sys.meta_path)
    _api_finder_installed = any(isinstance(f, ApiSubmoduleFinder) for f in sys.meta_path)
    
    if not _selection_finder_installed:
        sys.meta_path.insert(0, SelectionSubmoduleFinder())
    if not _benchmarking_finder_installed:
        sys.meta_path.insert(0, BenchmarkingSubmoduleFinder())
    if not _conversion_finder_installed:
        sys.meta_path.insert(0, ConversionSubmoduleFinder())
    if not _api_finder_installed:
        sys.meta_path.insert(0, ApiSubmoduleFinder())
except Exception:
    # If finder installation fails, continue - the shims will handle it when modules are imported
    pass

from infrastructure.paths import resolve_output_path

# Global variable to store the log file path and TeeOutput instance
_pytest_log_file = None
_pytest_tee = None


def pytest_configure(config):
    """Configure pytest hooks."""
    global _pytest_log_file

    # Check environment for torch-requiring tests
    import os
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    
    # Warn if torch tests are being run without the correct environment
    if config.option.markexpr and ("torch" in config.option.markexpr or "requires_torch" in config.option.markexpr):
        if conda_env != "resume-ner-training":
            print(f"\n{'='*60}")
            print(f"WARNING: Torch-requiring tests detected but current environment is '{conda_env}'")
            print(f"Expected environment: 'resume-ner-training'")
            print(f"Activate with: conda activate resume-ner-training")
            print(f"{'='*60}\n")

    # Create timestamped log file for this test run
    log_file = config.getoption("--log-file", default=None)

    if log_file:
        log_path = Path(log_file).resolve()
    else:
        # Create timestamped log file in outputs/pytest_logs directory (centralized)
        config_dir = ROOT_DIR / "config"
        log_dir = resolve_output_path(ROOT_DIR, config_dir, "pytest_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"pytest_{timestamp}.log"

    _pytest_log_file = log_path

    # Set the log file option for pytest's built-in logging
    if not config.getoption("--log-file"):
        config.option.log_file = str(log_path)

    # Print log file location
    print(f"\n{'='*60}")
    print(f"[INFO] Pytest log file: {log_path}")
    print(f"{'='*60}\n")


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    global _pytest_tee, _pytest_log_file
    
    # Check environment for torch-requiring tests
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    skip_torch = pytest.mark.skip(reason="Requires resume-ner-training environment. Run: conda activate resume-ner-training")
    
    for item in items:
        # Check if test requires torch
        torch_markers = [mark for mark in item.iter_markers() if mark.name in ("torch", "requires_torch")]
        
        if torch_markers and conda_env != "resume-ner-training":
            # Skip test if it requires torch but we're not in the right environment
            item.add_marker(skip_torch)

    if _pytest_tee and _pytest_log_file:
        try:
            _pytest_tee.write(
                f"\n[COLLECTION] Collected {len(items)} item(s)\n")
            for item in items:
                _pytest_tee.write(f"  - {item.nodeid}\n")
            _pytest_tee.flush()
        except Exception:
            pass


def pytest_sessionstart(session):
    """Setup logging at the start of the test session."""
    global _pytest_log_file, _pytest_tee

    if _pytest_log_file:
        try:
            _pytest_log_file.parent.mkdir(parents=True, exist_ok=True)

            # Open log file for writing (write mode to start fresh)
            _pytest_tee = open(_pytest_log_file, "w", encoding="utf-8")

            # Write session start header to log file
            import platform
            import sys
            _pytest_tee.write(f"{'='*60}\n")
            _pytest_tee.write(f"Pytest test session started\n")
            _pytest_tee.write(f"Log file: {_pytest_log_file}\n")
            _pytest_tee.write(f"Platform: {platform.platform()}\n")
            _pytest_tee.write(f"Python: {sys.version}\n")
            _pytest_tee.write(f"Rootdir: {session.config.rootdir}\n")
            _pytest_tee.write(f"{'='*60}\n\n")
            _pytest_tee.flush()

        except Exception as e:
            print(
                f"[WARNING] Could not setup file logging to {_pytest_log_file}: {e}")


def pytest_runtest_logstart(nodeid, location):
    """Log when a test starts."""
    global _pytest_tee, _pytest_log_file

    if _pytest_tee and _pytest_log_file:
        try:
            _pytest_tee.write(f"\n[TEST START] {nodeid}\n")
            _pytest_tee.flush()
        except Exception:
            pass


def pytest_runtest_logreport(report):
    """Log test results and output to file."""
    global _pytest_tee, _pytest_log_file

    if _pytest_tee and _pytest_log_file:
        try:
            # Write test result to log file for all phases
            status = report.outcome.upper()
            test_name = report.nodeid
            when = report.when
            duration = getattr(report, 'duration', 0)

            _pytest_tee.write(f"\n[TEST {when.upper()}] {status}: {test_name}")
            if duration:
                _pytest_tee.write(f" (duration: {duration:.2f}s)")
            _pytest_tee.write("\n")

            # Write captured output if available (only for call phase)
            if when == "call":
                if hasattr(report, 'capstdout') and report.capstdout:
                    _pytest_tee.write(f"[STDOUT]\n{report.capstdout}\n")
                if hasattr(report, 'capstderr') and report.capstderr:
                    _pytest_tee.write(f"[STDERR]\n{report.capstderr}\n")
                if hasattr(report, 'longrepr') and report.longrepr:
                    _pytest_tee.write(f"[ERROR]\n{str(report.longrepr)}\n")

            _pytest_tee.flush()
        except Exception:
            pass


def pytest_collectreport(report):
    """Log collection reports, including errors."""
    global _pytest_tee, _pytest_log_file

    if _pytest_tee and _pytest_log_file:
        try:
            if report.failed:
                _pytest_tee.write(f"\n{'='*60}\n")
                _pytest_tee.write(f"[COLLECTION ERROR] {report.nodeid}\n")
                _pytest_tee.write(f"{'='*60}\n")

                # Write full error details
                if hasattr(report, 'longrepr') and report.longrepr:
                    error_str = str(report.longrepr)
                    # Format the error nicely
                    _pytest_tee.write(f"{error_str}\n")
                elif hasattr(report, 'longreprtext') and report.longreprtext:
                    _pytest_tee.write(f"{report.longreprtext}\n")

                # Also try to get exception info
                if hasattr(report, 'exception') and report.exception:
                    import traceback
                    _pytest_tee.write(f"\nException:\n")
                    _pytest_tee.write(
                        f"{''.join(traceback.format_exception(type(report.exception), report.exception, report.exception.__traceback__))}\n")

                _pytest_tee.write(f"{'='*60}\n")
                _pytest_tee.flush()
        except Exception as e:
            # Fallback: at least log that we tried
            try:
                _pytest_tee.write(
                    f"\n[COLLECTION ERROR] {report.nodeid} (error details could not be captured: {e})\n")
                _pytest_tee.flush()
            except:
                pass


def pytest_sessionfinish(session, exitstatus):
    """Cleanup after pytest session."""
    global _pytest_tee, _pytest_log_file

    if _pytest_tee and _pytest_log_file:
        try:
            # Get test summary from terminal reporter if available
            try:
                reporter = session.config.pluginmanager.get_plugin(
                    'terminalreporter')
                if reporter:
                    stats = reporter.stats
                    passed = len(stats.get('passed', []))
                    failed = len(stats.get('failed', []))
                    skipped = len(stats.get('skipped', []))
                    error = len(stats.get('error', []))

                    _pytest_tee.write(f"\n[SUMMARY]\n")
                    _pytest_tee.write(f"  Passed: {passed}\n")
                    _pytest_tee.write(f"  Failed: {failed}\n")
                    _pytest_tee.write(f"  Skipped: {skipped}\n")
                    _pytest_tee.write(f"  Errors: {error}\n")
            except Exception:
                pass

            # Write session end footer to log file
            _pytest_tee.write(f"\n{'='*60}\n")
            _pytest_tee.write(
                f"Pytest test session finished (exit status: {exitstatus})\n")
            _pytest_tee.write(f"{'='*60}\n")
            _pytest_tee.flush()
        except Exception:
            pass

    if _pytest_tee:
        try:
            _pytest_tee.close()
        except Exception:
            pass
        _pytest_tee = None


def pytest_unconfigure(config):
    """Called after all tests are collected and run, and after sessionfinish.

    This is the last hook to run, so it's the perfect place to print the log file
    location at the very end, after all pytest output including error details.
    """
    global _pytest_log_file

    # Print log file location at the very end, after all pytest output
    # This hook runs after pytest_sessionfinish and all error output
    if _pytest_log_file:
        print(f"\n{'='*60}")
        print(f"[INFO] Pytest log file: {_pytest_log_file}")
        print(f"{'='*60}\n")

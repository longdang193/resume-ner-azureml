"""Global pytest configuration and fixtures.

This file is automatically discovered by pytest and applies to all tests.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root and src to Python path for all tests
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Global variable to store the log file path and TeeOutput instance
_pytest_log_file = None
_pytest_tee = None


def pytest_configure(config):
    """Configure pytest hooks."""
    global _pytest_log_file

    # Create timestamped log file for this test run
    log_file = config.getoption("--log-file", default=None)

    if log_file:
        log_path = Path(log_file).resolve()
    else:
        # Create timestamped log file in outputs/pytest_logs directory
        log_dir = ROOT_DIR / "outputs" / "pytest_logs"
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
    """Log collection information."""
    global _pytest_tee, _pytest_log_file

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

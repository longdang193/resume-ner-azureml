#!/usr/bin/env python3
"""HPO Pipeline Test Script with Tiny Datasets.

This script validates that the HPO (Hyperparameter Optimization) pipeline works correctly
with different tiny datasets. It's a standalone version of the test_hpo_with_tiny_datasets.ipynb notebook.

The script tests:
1. HPO Pipeline Completion: Verify HPO sweeps complete successfully with tiny datasets
2. K-Fold Cross-Validation: Test k-fold CV with small datasets and edge cases
3. Edge Cases: Test minimal k, small validation sets, batch size issues
4. Random Seed Variants: Test multiple random seed variants (seed0, seed1, seed2, ...)

Usage:
    python tests/e2e/test_hpo_with_tiny_datasets.py [--seeds 0 1 2] [--backbones distilbert distilroberta] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path


class TeeOutput:
    """Capture stdout/stderr and write to both console and log file."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.file_handle = open(log_file, "a", encoding="utf-8")
        
    def write(self, message: str):
        """Write to both console and log file."""
        # Write to original stdout (console)
        self.original_stdout.write(message)
        self.original_stdout.flush()
        # Also write to log file (raw output, no timestamp duplication)
        if self.file_handle is not None:
            self.file_handle.write(message)
            self.file_handle.flush()
        
    def flush(self):
        """Flush both outputs."""
        self.original_stdout.flush()
        if self.file_handle is not None:
            self.file_handle.flush()
        
    def close(self):
        """Close the log file and restore original stdout/stderr."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # tests/e2e/ -> tests/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tests.integration.orchestrators.test_orchestrator import (
    test_deterministic_hpo_multiple_backbones,
    test_random_seed_variants_multiple_backbones,
    test_kfold_validation,
    test_edge_case_k_too_large,
    test_edge_cases_suite,
)
from tests.integration.aggregators.result_aggregator import (
    collect_test_results,
    build_test_details,
)
from tests.integration.comparators.result_comparator import compare_results
from tests.integration.setup.environment_setup import setup_test_environment
from tests.fixtures.config.test_config_loader import get_test_config, BACKBONES_LIST
from tests.fixtures.hpo_test_helpers import (
    DEFAULT_RANDOM_SEED,
    MINIMAL_K_FOLDS,
    DEFAULT_BACKBONE,
    METRIC_DECIMAL_PLACES,
    SEPARATOR_WIDTH,
    VERY_SMALL_VALIDATION_THRESHOLD,
    print_test_summary,
    print_comparison,
    print_hpo_results,
    print_kfold_results,
    print_edge_case_k_too_large_results,
    print_edge_case_results,
)


def setup_logging(output_dir: Path, log_file: Path = None) -> tuple[logging.Logger, Path]:
    """
    Setup logging to both console and file.
    
    Args:
        output_dir: Directory where log file will be created (if log_file not provided)
        log_file: Optional specific log file path
        
    Returns:
        Tuple of (logger, log_file_path)
    """
    if log_file is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"test_hpo_{timestamp}.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,  # Override any existing configuration
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    return logger, log_file


def main():
    """Main entry point for HPO pipeline tests."""
    parser = argparse.ArgumentParser(
        description="Test HPO pipeline with tiny datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python tests/e2e/test_hpo_with_tiny_datasets.py

  # Test specific seeds
  python tests/e2e/test_hpo_with_tiny_datasets.py --seeds 0 1 2

  # Test specific backbones
  python tests/e2e/test_hpo_with_tiny_datasets.py --backbones distilbert

  # Custom output directory
  python tests/e2e/test_hpo_with_tiny_datasets.py --output-dir outputs/my_tests
        """,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to test (default: from config/test/hpo_pipeline.yaml)",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        default=None,
        help="Backbones to test (default: from config/test/hpo_pipeline.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test results (default: from config)",
    )
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Skip deterministic HPO test (seed0 dataset)",
    )
    parser.add_argument(
        "--skip-random-seeds",
        action="store_true",
        help="Skip random seed variants test",
    )
    parser.add_argument(
        "--skip-kfold",
        action="store_true",
        help="Skip k-fold validation test",
    )
    parser.add_argument(
        "--skip-edge-cases",
        action="store_true",
        help="Skip edge cases tests",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: auto-generated in output directory)",
    )

    args = parser.parse_args()

    try:
        # Setup paths
        root_dir = PROJECT_ROOT
        
        # Determine output directory early for logging setup
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Get default from config
            test_config = get_test_config(root_dir)
            output_section = test_config.get("output", {})
            output_dir = root_dir / output_section.get("base_dir", "outputs/hpo_tests")
        
        # Setup logging
        if args.log_file:
            log_file_path = Path(args.log_file)
            logger, actual_log_file = setup_logging(output_dir, log_file_path)
        else:
            logger, actual_log_file = setup_logging(output_dir)
        
        logger.info("=" * 60)
        logger.info("HPO Pipeline Test with Tiny Datasets")
        logger.info("=" * 60)
        logger.info(f"Project root: {root_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Log file: {actual_log_file}")
        
        # Setup TeeOutput to capture all print statements to log file
        # This will capture all stdout/stderr output to the log file
        tee = TeeOutput(actual_log_file)
        sys.stdout = tee
        sys.stderr = tee
        
        print(f"Project root: {root_dir}")
        print(f"Log file: {actual_log_file}")

        # Load configuration from config/test/hpo_pipeline.yaml
        test_config = get_test_config(root_dir)
        datasets_section = test_config.get("datasets", {})
        configs_section = test_config.get("configs", {})

        # Get random seeds to test (from args or config)
        if args.seeds is not None:
            random_seeds_to_test = args.seeds
        else:
            random_seeds_to_test = datasets_section.get("random_seeds", [0])

        # Get backbones to test (from args or config)
        if args.backbones is not None:
            backbones_to_test = args.backbones
        else:
            backbones_to_test = BACKBONES_LIST

        logger.info(f"Configuration loaded from: config/test/hpo_pipeline.yaml")
        logger.info(f"Random seeds to test: {random_seeds_to_test}")
        logger.info(f"Backbones to test: {backbones_to_test}")
        logger.info(f"Default backbone: {DEFAULT_BACKBONE}")
        logger.info(f"Default random seed: {DEFAULT_RANDOM_SEED}")
        
        print(f"Configuration loaded from: config/test/hpo_pipeline.yaml")
        print(f"Random seeds to test: {random_seeds_to_test}")
        print(f"Backbones to test: {backbones_to_test}")
        print(f"Default backbone: {DEFAULT_BACKBONE}")
        print(f"Default random seed: {DEFAULT_RANDOM_SEED}")

        # Setup test environment: load configs, setup paths, initialize MLflow
        # Note: output_dir may be overridden by setup_test_environment, but logging is already set up
        env = setup_test_environment(root_dir=root_dir, output_dir=output_dir)

        config_dir = env["config_dir"]
        hpo_config = env["hpo_config"]
        train_config = env["train_config"]
        output_dir = env["output_dir"]
        deterministic_dataset = env["deterministic_dataset"]
        mlflow_tracking_uri = env["mlflow_tracking_uri"]

        print(f"\n{'=' * 60}")
        print("Test Environment Configuration")
        print(f"{'=' * 60}")
        print(f"Config directory: {config_dir}")
        print(f"HPO config file: {hpo_config.get('_config_path', 'N/A')}")
        print(f"  Max trials: {hpo_config.get('sampling', {}).get('max_trials', 'N/A')}")
        print(f"  K-folds: {hpo_config.get('k_fold', {}).get('n_splits', 'N/A')}")
        print(
            f"  Objective: {hpo_config.get('objective', {}).get('metric', 'N/A')} "
            f"({hpo_config.get('objective', {}).get('goal', 'N/A')})"
        )
        print(f"Train config file: {train_config.get('_config_path', 'N/A')}")
        print(f"Output directory: {output_dir}")
        print(f"MLflow tracking URI: {mlflow_tracking_uri}")
        print(f"Dataset base path: {deterministic_dataset}")
        print(f"Random seeds to test: {random_seeds_to_test}")
        print(f"Backbones to test: {backbones_to_test}")
        print(f"Note: All datasets are seed-based (seed0, seed1, seed2, ...)")
        print(f"{'=' * 60}\n")

        # Initialize result storage
        seed0_results = None
        seed0_results_by_backbone = {}
        random_seed_results = None
        random_seed_results_by_backbone = {}
        kfold_results = None
        k_too_large_results = None
        edge_case_results = None

        # Test Suite 1: HPO with Seed0 Dataset (Multiple Backbones)
        if not args.skip_deterministic:
            print(f"\n{'=' * 60}")
            print("Test Suite 1: HPO with Seed0 Dataset (Multiple Backbones)")
            print(f"{'=' * 60}")
            seed0_dataset = deterministic_dataset / "seed0"

            seed0_results_by_backbone = test_deterministic_hpo_multiple_backbones(
                dataset_path=seed0_dataset,
                config_dir=config_dir,
                hpo_config=hpo_config,
                train_config=train_config,
                output_dir=output_dir,
                backbones=backbones_to_test,
            )

            # Print results for each backbone
            for backbone, results in seed0_results_by_backbone.items():
                if results:
                    print_hpo_results(results, f"Seed0 Dataset HPO Results ({backbone})")
                    print()

            # For backward compatibility, use first backbone's results
            seed0_results = (
                seed0_results_by_backbone.get(backbones_to_test[0])
                if backbones_to_test
                else None
            )
        else:
            print("\nSkipping Test Suite 1: HPO with Seed0 Dataset (--skip-deterministic)")

        # Test Suite 2: HPO with Random Seed Variants
        if not args.skip_random_seeds:
            print(f"\n{'=' * 60}")
            print("Test Suite 2: HPO with Random Seed Variants")
            print(f"{'=' * 60}")

            random_seed_results_by_backbone = test_random_seed_variants_multiple_backbones(
                dataset_base_path=deterministic_dataset,
                seeds=random_seeds_to_test,
                config_dir=config_dir,
                hpo_config=hpo_config,
                train_config=train_config,
                output_dir=output_dir,
                backbones=backbones_to_test,
            )

            # Print results for each backbone and seed
            for backbone, backbone_results in random_seed_results_by_backbone.items():
                if backbone_results:
                    print(f"\n{'=' * 60}")
                    print(f"Backbone: {backbone}")
                    print("=" * 60)
                    for seed, seed_results in backbone_results.items():
                        print_hpo_results(
                            seed_results, f"Random Seed Variant (seed {seed}) HPO Results"
                        )

            # For backward compatibility, use first backbone's results
            random_seed_results = (
                random_seed_results_by_backbone.get(backbones_to_test[0])
                if backbones_to_test
                else None
            )

            # Compare seed0 vs other random seed results (for first backbone)
            if seed0_results and random_seed_results:
                comparison = compare_results(seed0_results, random_seed_results)
                if comparison:
                    print(f"\n{'=' * 60}")
                    print("Comparison: Deterministic vs Random Seed Variants")
                    print(f"{'=' * 60}")
                    print_comparison(comparison)
        else:
            print("\nSkipping Test Suite 2: Random Seed Variants (--skip-random-seeds)")

        # Test Suite 3: K-Fold Cross-Validation Validation
        if not args.skip_kfold:
            print(f"\n{'=' * 60}")
            print("Test Suite 3: K-Fold Cross-Validation Validation")
            print(f"{'=' * 60}")

            seed0_dataset = deterministic_dataset / "seed0"
            kfold_results = test_kfold_validation(
                dataset_path=seed0_dataset,
                hpo_config=hpo_config,
            )

            # Print results
            if kfold_results:
                k_configured = hpo_config.get("k_fold", {}).get("n_splits", 3)
                print_kfold_results(kfold_results, k_configured)
        else:
            print("\nSkipping Test Suite 3: K-Fold Validation (--skip-kfold)")

        # Test Suite 4: Edge Case - k > n_samples
        if not args.skip_edge_cases:
            print(f"\n{'=' * 60}")
            print("Test Suite 4: Edge Case - k > n_samples")
            print(f"{'=' * 60}")

            seed0_dataset = deterministic_dataset / "seed0"
            k_too_large_results = test_edge_case_k_too_large(dataset_path=seed0_dataset)

            # Print results
            if k_too_large_results:
                print_edge_case_k_too_large_results(k_too_large_results)
        else:
            print("\nSkipping Test Suite 4: Edge Case k > n_samples (--skip-edge-cases)")

        # Test Suite 5: Edge Cases Suite
        if not args.skip_edge_cases:
            print(f"\n{'=' * 60}")
            print("Test Suite 5: Edge Cases Suite")
            print(f"{'=' * 60}")

            seed0_dataset = deterministic_dataset / "seed0"
            edge_case_results = test_edge_cases_suite(
                dataset_path=seed0_dataset,
                hpo_config=hpo_config,
                train_config=train_config,
            )

            # Print results
            if edge_case_results:
                print_edge_case_results(edge_case_results)
        else:
            print("\nSkipping Test Suite 5: Edge Cases Suite (--skip-edge-cases)")

        # Aggregate all test results
        print(f"\n{'=' * 60}")
        print("Summary: Test Results")
        print(f"{'=' * 60}")

        test_summary = collect_test_results(
            deterministic_results=seed0_results,
            random_seed_results=random_seed_results,
            kfold_results=kfold_results,
            k_too_large_results=k_too_large_results,
            edge_case_results=edge_case_results,
        )

        test_details = build_test_details(
            deterministic_results=seed0_results,
            random_seed_results=random_seed_results,
            kfold_results=kfold_results,
        )

        # Print final summary
        print_test_summary(test_summary=test_summary, test_details=test_details)

        # Calculate overall success from test_summary
        # Filter out None values (tests that weren't run)
        all_tests = [v for v in test_summary.values() if v is not None]
        overall_success = all(all_tests) if all_tests else False
        if overall_success:
            logger.info("=" * 60)
            logger.info("✓ All tests passed!")
            logger.info("=" * 60)
            print(f"\n{'=' * 60}")
            print("✓ All tests passed!")
            print(f"{'=' * 60}")
            print(f"\nLog file saved to: {actual_log_file}")
            # Restore stdout/stderr before returning
            if 'tee' in locals():
                tee.close()
            return 0
        else:
            logger.error("=" * 60)
            logger.error("✗ Some tests failed. Review the detailed output above.")
            logger.error("=" * 60)
            print(f"\n{'=' * 60}")
            print("✗ Some tests failed. Review the detailed output above.")
            print(f"{'=' * 60}")
            print(f"\nLog file saved to: {actual_log_file}")
            # Restore stdout/stderr before returning
            if 'tee' in locals():
                tee.close()
            return 1

    except Exception as e:
        # Try to log if logger exists, otherwise just print
        try:
            logger.exception("Test execution failed")
            logger.error(f"Exception: {e}")
        except:
            pass
        print(f"\n{'=' * 60}")
        print(f"✗ Test execution failed: {e}")
        print(f"{'=' * 60}")
        try:
            print(f"\nLog file saved to: {actual_log_file}")
        except:
            pass
        import traceback
        traceback.print_exc()
        # Restore stdout/stderr before returning
        if 'tee' in locals():
            tee.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())


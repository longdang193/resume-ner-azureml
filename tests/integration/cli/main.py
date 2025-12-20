"""CLI interface for HPO pipeline tests.

This module is responsible solely for parsing command-line arguments and
invoking the test orchestrator. It contains no business logic or orchestration.
"""

import argparse
from pathlib import Path

from tests.fixtures.config.test_config_loader import get_test_config
from tests.fixtures.presenters.result_formatters import (
    print_comparison,
    print_edge_case_k_too_large_results,
    print_edge_case_results,
    print_hpo_results,
    print_kfold_results,
    print_test_summary,
)
from tests.integration.orchestrators.test_orchestrator import (
    run_all_tests,
    test_deterministic_hpo,
    test_edge_case_k_too_large,
    test_edge_cases_suite,
    test_kfold_validation,
    test_random_seed_variants,
)
from tests.integration.setup.environment_setup import setup_test_environment


def main():
    """Command-line interface for running HPO pipeline tests."""
    parser = argparse.ArgumentParser(
        description="Run HPO pipeline tests with tiny datasets"
    )

    script_dir = Path(__file__).parent
    default_root_dir = script_dir.parent.parent.parent

    # Load config to get defaults for help text
    try:
        test_config = get_test_config(default_root_dir)
        configs_section = test_config.get("configs", {})
        datasets_section = test_config.get("datasets", {})
        output_section = test_config.get("output", {})

        default_seeds = datasets_section.get("random_seeds", [0])
        default_hpo_config = configs_section.get("hpo_config", "hpo/smoke.yaml")
        default_train_config = configs_section.get("train_config", "train.yaml")
        default_output_dir = output_section.get("base_dir", "outputs/hpo_tests")
    except Exception:
        # Fallback if config not available
        default_seeds = [0]
        default_hpo_config = "hpo/smoke.yaml"
        default_train_config = "train.yaml"
        default_output_dir = "outputs/hpo_tests"

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=default_seeds,
        help=f"List of seed numbers to test (default: {default_seeds})",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=default_root_dir,
        help=f"Project root directory (default: {default_root_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Test output directory (default: {default_output_dir})",
    )
    parser.add_argument(
        "--hpo-config",
        type=Path,
        help=f"Path to HPO config file (default: config/{default_hpo_config})",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        help=f"Path to train config file (default: config/{default_train_config})",
    )
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Skip deterministic dataset test",
    )
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip random seed variant tests",
    )
    parser.add_argument(
        "--skip-kfold",
        action="store_true",
        help="Skip k-fold validation tests",
    )
    parser.add_argument(
        "--skip-edge-cases",
        action="store_true",
        help="Skip edge case tests",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        output_path = args.output_dir or (args.root_dir / default_output_dir)
        print("=" * 60)
        print("Test Configuration")
        print("=" * 60)
        print(f"Root directory: {args.root_dir}")
        print(f"Seeds to test: {args.seeds}")
        print(f"HPO config: {args.hpo_config or f'config/{default_hpo_config}'}")
        print(f"Train config: {args.train_config or f'config/{default_train_config}'}")
        print(f"Output directory: {output_path}")
        print("=" * 60)

    results = run_all_tests(
        root_dir=args.root_dir,
        random_seeds=args.seeds,
        hpo_config_path=args.hpo_config,
        train_config_path=args.train_config,
        output_dir=args.output_dir,
        skip_deterministic=args.skip_deterministic,
        skip_random=args.skip_random,
        skip_kfold=args.skip_kfold,
        skip_edge_cases=args.skip_edge_cases,
    )

    # Presentation layer: print results
    if results.get("deterministic_results"):
        print_hpo_results(
            results["deterministic_results"],
            "Deterministic Dataset HPO Results"
        )

    if results.get("random_seed_results"):
        for seed, seed_results in results["random_seed_results"].items():
            print_hpo_results(
                seed_results,
                f"Random Seed Variant (seed {seed}) HPO Results"
            )

    if results.get("comparison"):
        print_comparison(results["comparison"])

    if results.get("kfold_results"):
        k_configured = None
        # Try to get k from results or environment
        env = setup_test_environment(root_dir=args.root_dir)
        if env.get("hpo_config"):
            k_configured = env["hpo_config"].get("k_fold", {}).get("n_splits")
        if k_configured is None:
            k_configured = 3  # Default fallback
        print_kfold_results(results["kfold_results"], k_configured)

    if results.get("k_too_large_results"):
        print_edge_case_k_too_large_results(results["k_too_large_results"])

    if results.get("edge_case_results"):
        print_edge_case_results(results["edge_case_results"])

    print_test_summary(
        test_summary=results["test_summary"],
        test_details=results["test_details"],
    )

    # Exit with error code if any tests failed
    all_tests = [v for v in results["test_summary"].values() if v is not None]
    if all_tests and not all(all_tests):
        exit(1)


if __name__ == "__main__":
    main()


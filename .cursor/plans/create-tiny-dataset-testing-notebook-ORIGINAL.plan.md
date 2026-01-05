# Create Testing Notebook for Tiny Datasets

## Original Plan

Create a testing notebook (`tests/test_hpo_with_tiny_datasets.ipynb`) to validate the HPO (Hyperparameter Optimization) pipeline works correctly with small datasets created by `tests/00_make_tiny_dataset.ipynb`.

## Purpose

- **Catch pipeline issues early**: Test HPO pipeline with small datasets before running on full data
- **Validate k-fold CV**: Ensure cross-validation works correctly with very small datasets (8 samples, 3 folds)
- **Test edge cases**: Verify pipeline handles minimal configurations gracefully
- **Compare datasets**: Test both deterministic and random seed variants

## Test Areas

1. **HPO Pipeline Completion**: Verify HPO sweeps complete successfully with tiny datasets

- Test with deterministic dataset (`dataset_tiny/`)
- Test with random seed variants (`dataset_tiny/seed{N}/`)

2. **K-Fold Cross-Validation**: Test k-fold CV with small datasets and edge cases

- Validate splits are created correctly
- Check that all samples are used
- Verify no overlapping train/val indices
- Test edge case: k > n_samples (should fail gracefully)

3. **Edge Cases**: Test minimal k, small validation sets, batch size issues

- Minimal k (k=2)
- Batch size vs validation set size
- Very small validation sets (≤2 samples)

## Implementation Steps

1. Create notebook structure with markdown cells explaining purpose
2. Setup: Import dependencies and configure paths
3. Test Suite 1: HPO Completion with Deterministic Dataset
4. Test Suite 2: HPO Completion with Random Seed Variant(s)
5. Test Suite 3: K-Fold Cross-Validation Validation
6. Test Suite 4: Edge Case Testing
7. Summary: Aggregate and display test results

## Configuration

- Use `config/hpo/smoke.yaml` for HPO config (small number of trials for fast testing)
- Use `config/train.yaml` for training config
- Test with `dataset_tiny/` (deterministic) and `dataset_tiny/seed{N}/` (random variants)
- Configurable `RANDOM_SEEDS_TO_TEST` list (e.g., `[0]` or `[0, 1, 2]`)

## Prerequisites

- Run `tests/00_make_tiny_dataset.ipynb` first to create test datasets
- Ensure `dataset_tiny/` and `dataset_tiny/seed{N}/` directories exist
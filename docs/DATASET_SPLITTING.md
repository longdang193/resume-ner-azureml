<!-- Dataset splitting guide for Resume NER -->

# Dataset Splitting

## Goals

- Create a held-out **test** set (never used in HPO/training).
- Use **stratified k-fold CV** on the training set to handle class imbalance.
- Keep **backward compatibility** (splitting is opt-in via config).

## Default Strategy

1. Initial split: 80/20 train/test (configurable).
2. Stratified by entity presence (per document) when enabled.
3. K-fold CV on the **train** portion during HPO.
4. Final evaluation only on the **test** set.

## Configuration

`config/data/*.yaml`
```yaml
splitting:
  train_test_ratio: 0.8
  stratified: true
  random_seed: 42
```

`config/hpo/*.yaml`
```yaml
k_fold:
  enabled: true
  n_splits: 5
  stratified: true
  random_seed: 42
  shuffle: true
```

## API Reference

- `split_train_test(dataset, train_ratio, stratified, random_seed, entity_types)`
  - Splits dataset into train/test with optional stratification.
- `save_split_files(output_dir, train_data, test_data)`
  - Writes `train.json` and `test.json`.
- `create_kfold_splits(dataset, k, random_seed, shuffle, stratified)`
  - Generates folds; falls back to non-stratified if labels are degenerate.
- `validate_splits(dataset, splits, entity_types)`
  - Prints per-fold entity counts to spot missing classes.

## Validation Checklist

- Test set exists (`test.json`) and is excluded from HPO/training.
- Each CV fold contains examples of rare entities (EMAIL, PHONE, etc.).
- Splits are reproducible (fixed `random_seed`, saved split files).



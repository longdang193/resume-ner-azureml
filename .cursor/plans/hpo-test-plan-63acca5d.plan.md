<!-- 63acca5d-0475-4c97-9ccf-7cb3d0a8af2c a10490f3-7f6c-48ef-b148-5c1442ccc57e -->
# HPO Process Test Plan

## 1. Scope and Assumptions

### Scope

- Test the complete HPO orchestration workflow from config loading through best trial selection
- Validate all scenarios explicitly defined in `config/hpo/smoke.yaml`
- Ensure compatibility with execution flow in `notebooks/01_orchestrate_training.ipynb`
- Test across multiple environments (local, colab, kaggle, azureml)
- Validate MLflow structure, file system outputs, cache files, and resume behavior

### Assumptions

- Tests run in isolated temporary directories
- MLflow tracking can be mocked or use local SQLite backend
- Azure ML services are mocked (no real workspace access required)
- Google Drive sync is mocked for Colab environment tests
- Test datasets are minimal (tiny datasets) to keep execution time reasonable
- Config files follow the schema defined in `config/` directory

### Out of Scope

- Actual model training (mocked or skipped)
- Real Azure ML workspace operations (mocked)
- Real Google Drive operations (mocked)
- Performance/load testing
- Multi-GPU distributed training (single process only)

## 2. Inventory of HPO Scenarios

### Scenarios from smoke.yaml

| Scenario | Notebook Step(s) | Test Level | Expected Artifacts/Tags/Paths |
|----------|------------------|------------|-------------------------------|
| **Search Space: backbone=distilbert** | Config load → Study creation | Unit, Component | `code.backbone=distilbert` tag, study name includes backbone |
| **Search Space: learning_rate (loguniform 1e-5 to 5e-5)** | Trial sampling → Training | Unit, Component | Param logged, value in range |
| **Search Space: batch_size (choice [4])** | Trial sampling → Training | Unit, Component | Param logged, value=4 |
| **Search Space: dropout (uniform 0.1-0.3)** | Trial sampling → Training | Unit, Component | Param logged, value in range |
| **Search Space: weight_decay (loguniform 0.001-0.1)** | Trial sampling → Training | Unit, Component | Param logged, value in range |
| **Sampling: algorithm=random, max_trials=1** | Study creation → Trial execution | Component, Integration | Exactly 1 trial run, random sampling |
| **Sampling: timeout_minutes=20** | Study creation | Component | Study timeout set correctly |
| **Checkpoint: enabled=true, auto_resume=true** | Study creation → Resume | Component, Integration | `study.db` created, resume works |
| **Checkpoint: study_name template with {backbone}** | Study creation | Component | Study name resolves to `hpo_distilbert_smoke_test_path_testing_23` |
| **Checkpoint: storage_path template with {study_name}** | Study creation | Component | Storage path resolves `{study_name}/study.db` to `hpo_distilbert_smoke_test_path_testing_23/study.db` |
| **Checkpoint: save_only_best=true** | Trial completion → Cleanup | Component, Integration | Only best trial checkpoint saved |
| **MLflow: log_best_checkpoint=true** | Best trial selection → Artifact logging | Component, Integration | Best checkpoint logged as MLflow artifact |
| **Early termination: bandit policy** | Trial execution | Component | Pruning works, trials marked PRUNED |
| **Early termination: evaluation_interval=1** | Trial execution | Component | Pruning checked every trial |
| **Early termination: slack_factor=0.2** | Trial execution | Component | Pruning threshold calculated correctly |
| **Early termination: delay_evaluation=2** | Trial execution | Component | First 2 trials not pruned |
| **Objective: metric=macro-f1, goal=maximize** | Trial execution → Selection | Component, Integration | Metric logged, best trial has highest value |
| **Selection: accuracy_threshold=0.015** | Best trial selection | Component, Integration | Speed tradeoff applied when within threshold |
| **Selection: use_relative_threshold=true** | Best trial selection | Component | Relative threshold calculation correct |
| **Selection: min_accuracy_gain=0.02** | Best trial selection | Component | Slower model only selected if >2% better |
| **k_fold: enabled=true, n_splits=2** | Trial execution | Component, Integration | 2 fold runs created, CV metrics aggregated |
| **k_fold: random_seed=42, shuffle=true, stratified=true** | Fold creation | Unit, Component | Fold splits reproducible, stratified |
| **Refit: enabled=true** | Post-HPO | Component, Integration | Refit run created, trained on full data |
| **Cleanup: disable_auto_cleanup=false (cleanup enabled)** | Run setup | Component | Interrupted runs tagged with code.interrupted=true |
| **Cleanup: disable_auto_optuna_mark=false (marking enabled)** | Resume | Component | RUNNING trials marked as FAILED on resume |
| **All smoke.yaml settings applied together** | Full workflow | Integration | All settings validated simultaneously in single run |

### Additional Edge Cases

| Scenario | Test Level | Expected Behavior |
|----------|------------|-------------------|
| Missing checkpoint file on resume | Component | New study created (no resume) |
| Corrupted checkpoint file | Component | Error handled gracefully, new study created |
| MLflow tracking disabled | Component | Runs continue without MLflow |
| Invalid hyperparameter range | Unit | Validation error raised |
| Trial failure during training | Component | Trial marked FAILED, study continues |
| All trials pruned | Component | Study completes with no best trial |
| Refit disabled | Component | No refit run created |
| k_fold disabled | Component | Single training run per trial |
| Missing config keys (fallbacks) | Unit, Component | Default values used, no crashes |

## 3. Test Matrix

| Scenario | Environment | Storage Env | Config Variants | Expected Outputs |
|----------|------------|-------------|-----------------|------------------|
| Basic HPO (1 trial, no CV) | local | local | smoke.yaml (k_fold.enabled=false) | `outputs/hpo/local/distilbert/study-{hash}/trial-{hash}/checkpoint/`, MLflow parent run + 1 trial run |
| HPO with CV (2 folds) | local | local | smoke.yaml (default) | `outputs/hpo/local/distilbert/study-{hash}/trial-{hash}/cv/fold0/checkpoint/`, `cv/fold1/checkpoint/`, MLflow parent + trial + 2 fold runs |
| HPO with checkpoint resume | local | local | smoke.yaml (checkpoint.enabled=true) | `study.db` exists, resume works, no duplicate trials |
| HPO with refit | local | local | smoke.yaml (refit.enabled=true) | `outputs/hpo/.../trial-{hash}/refit/checkpoint/`, refit MLflow run |
| HPO in Colab | colab | colab | smoke.yaml | Paths use `/content/drive/MyDrive/...`, Drive sync works |
| HPO in Kaggle | kaggle | kaggle | smoke.yaml | Paths use `/kaggle/working/...` |
| HPO in AzureML | azureml | azureml | smoke.yaml | Paths use `/mnt/outputs/...`, Azure ML tracking |
| Best trial selection with speed tradeoff | local | local | smoke.yaml (selection.accuracy_threshold=0.015) | Faster model selected when within threshold |
| Early termination pruning | local | local | smoke.yaml (early_termination enabled) | Trials pruned after delay_evaluation=2 |
| Checkpoint save_only_best | local | local | smoke.yaml (checkpoint.save_only_best=true) | Only best trial checkpoint directory exists |

## 4. Detailed Test Cases

### 4.1 Unit Tests

#### Test: `test_search_space_translation`

**File**: `tests/unit/orchestration/test_hpo_search_space.py`

- **Setup**: Load smoke.yaml search_space section
- **Execution**: Call `translate_search_space_to_optuna()`
- **Assertions**:
- backbone choice values = ["distilbert"]
- learning_rate is loguniform with min=1e-5, max=5e-5
- batch_size is choice with values=[4]
- dropout is uniform with min=0.1, max=0.3
- weight_decay is loguniform with min=0.001, max=0.1
- **Teardown**: None

#### Test: `test_config_hash_computation`

**File**: `tests/unit/orchestration/test_config_loader.py`

- **Setup**: Sample config dicts
- **Execution**: Call `compute_config_hash()` for each domain
- **Assertions**:
- Hash length = CONFIG_HASH_LENGTH (16)
- Same config produces same hash
- Different configs produce different hashes
- Hash is deterministic (no randomness)
- **Teardown**: None

#### Test: `test_study_key_hash_computation`

**File**: `tests/unit/orchestration/test_mlflow_naming.py`

- **Setup**: Sample data_config, hpo_config, benchmark_config, backbone
- **Execution**: Call `build_hpo_study_key()` and `build_hpo_study_key_hash()`
- **Assertions**:
- Hash length = 32 (full SHA256) or 8 (short)
- Same inputs produce same hash
- Different inputs produce different hashes
- Hash includes all config components
- **Teardown**: None

#### Test: `test_trial_key_hash_computation`

**File**: `tests/unit/orchestration/test_mlflow_naming.py`

- **Setup**: study_key_hash, sample hyperparameters
- **Execution**: Call `build_hpo_trial_key()` and `build_hpo_trial_key_hash()`
- **Assertions**:
- Hash includes study_key_hash and hyperparameters
- Same trial params produce same hash
- Different params produce different hashes
- **Teardown**: None

#### Test: `test_run_name_generation`

**File**: `tests/unit/orchestration/test_naming_centralized.py`

- **Setup**: NamingContext for HPO trial, fold, refit, sweep
- **Execution**: Call `build_mlflow_run_name()` for each
- **Assertions**:
- HPO trial: `{env}_{model}_hpo_trial_study-{hash}_t{trial_number}`
- HPO fold: `{env}_{model}_hpo_trial_study-{hash}_t{trial_number}_fold{fold_idx}`
- HPO refit: `{env}_{model}_hpo_refit_study-{hash}_trial-{hash}_t{trial_number}`
- HPO sweep: `{env}_{model}_hpo_study-{hash}{semantic_suffix}`
- Names respect max_length=256, forbidden chars removed
- **Teardown**: None

#### Test: `test_path_building_v2`

**File**: `tests/unit/orchestration/test_paths.py`

- **Setup**: NamingContext with study_key_hash, trial_key_hash, storage_env
- **Execution**: Call `build_output_path()` for HPO
- **Assertions**:
- Path pattern: `outputs/hpo/{storage_env}/{model}/study-{study8}/trial-{trial8}`
- study8 = study_key_hash[:8], trial8 = trial_key_hash[:8]
- Path normalized (no invalid chars)
- **Teardown**: None

#### Test: `test_kfold_split_creation`

**File**: `tests/unit/training/test_cv_utils.py`

- **Setup**: Sample dataset, k=2, random_seed=42, shuffle=true, stratified=true
- **Execution**: Call `create_kfold_splits()`
- **Assertions**:
- Returns 2 folds
- All samples in exactly one fold
- Folds are stratified (class distribution similar)
- Same seed produces same splits
- Splits are saved to file correctly
- **Teardown**: Remove split file

#### Test: `test_selection_criteria`

**File**: `tests/unit/orchestration/test_best_trial_selection.py`

- **Setup**: Mock trial results with metrics and inference times
- **Execution**: Call selection logic with accuracy_threshold=0.015, use_relative_threshold=true
- **Assertions**:
- Faster model selected when accuracy within 1.5% (relative)
- Slower model selected when accuracy >1.5% better
- min_accuracy_gain=0.02 respected (slower only if >2% better)
- Tie-breaking deterministic
- **Teardown**: None

### 4.2 Component Tests

#### Test: `test_hpo_sweep_setup`

**File**: `tests/integration/hpo/test_hpo_sweep_setup.py`

- **Setup**: 
- tmp_path with config files (smoke.yaml)
- Mock MLflow client
- Tiny dataset
- **Execution**:
- Load configs via `load_experiment_config()` and `load_all_configs()`
- Call `setup_hpo_mlflow_run()`
- Create Optuna study with checkpoint
- **Assertions**:
- MLflow parent run created with correct tags (code.stage=hpo, code.project=resume-ner)
- Study name = `hpo_{backbone}_smoke_test_path_testing_23` (from checkpoint.study_name)
- Checkpoint file created at `{study_name}/study.db`
- study_key_hash and study_family_hash computed and tagged
- **Teardown**: Remove tmp_path

#### Test: `test_checkpoint_study_name_template_resolution`

**File**: `tests/integration/hpo/test_checkpoint_resolution.py`

- **Setup**: 
- Load smoke.yaml with checkpoint.study_name = "hpo_{backbone}_smoke_test_path_testing_23"
- Backbone = "distilbert"
- **Execution**: 
- Call `create_study_name()` with backbone="distilbert"
- **Assertions**:
- `{backbone}` placeholder replaced with "distilbert"
- Final study name = "hpo_distilbert_smoke_test_path_testing_23"
- Study name used consistently in Optuna study creation
- Study name matches MLflow run name (if checkpoint enabled)
- **Teardown**: None

#### Test: `test_checkpoint_storage_path_template_resolution`

**File**: `tests/integration/hpo/test_checkpoint_resolution.py`

- **Setup**: 
- Load smoke.yaml with checkpoint.storage_path = "{study_name}/study.db"
- Study name = "hpo_distilbert_smoke_test_path_testing_23"
- Output directory configured
- **Execution**: 
- Call `resolve_storage_path()` with checkpoint config
- **Assertions**:
- `{study_name}` placeholder replaced with actual study name
- Final storage path = `hpo_distilbert_smoke_test_path_testing_23/study.db`
- Path is relative to output directory
- Path created correctly when checkpoint saved
- **Teardown**: None

#### Test: `test_trial_execution_no_cv`

**File**: `tests/integration/hpo/test_trial_execution.py`

- **Setup**:
- Mock training subprocess (returns success, writes metrics.json)
- MLflow parent run active
- Optuna study created
- **Execution**:
- Run single trial (no CV) via `create_local_hpo_objective()`
- Trial samples hyperparameters
- Training subprocess called
- **Assertions**:
- Trial run created as child of parent
- Run name follows pattern from naming.yaml
- Hyperparameters logged as MLflow params
- Metrics logged (macro-f1)
- Trial folder created: `study-{hash}/trial-{hash}/checkpoint/`
- trial_meta.json written with study_key_hash, trial_key_hash, trial_number
- **Teardown**: Clean up MLflow runs

#### Test: `test_trial_execution_with_cv`

**File**: `tests/integration/hpo/test_trial_execution.py`

- **Setup**: Same as above, but k_fold.enabled=true, n_splits=2
- **Execution**: Run trial with CV
- **Assertions**:
- Trial run created
- 2 fold runs created as children of trial run
- Fold run names include `_fold0`, `_fold1`
- Fold outputs: `trial-{hash}/cv/fold0/checkpoint/`, `cv/fold1/checkpoint/`
- CV metrics aggregated: cv_mean, cv_std logged to trial run
- Individual fold metrics logged: `fold_0_macro-f1`, `fold_1_macro-f1`
- Fold splits saved to `fold_splits.json`
- **Teardown**: Clean up MLflow runs

#### Test: `test_checkpoint_resume`

**File**: `tests/integration/hpo/test_hpo_checkpoint_resume.py`

- **Setup**:
- Run HPO with max_trials=3, checkpoint enabled
- Interrupt after 2 trials (simulate)
- Checkpoint file exists
- **Execution**:
- Resume HPO with same study_name
- Call `run_local_hpo_sweep()` with should_resume=True
- **Assertions**:
- Study loads from checkpoint (not new study)
- Only 1 new trial runs (total 3)
- No duplicate trials
- Study state preserved (completed trials, pruned trials)
- **Teardown**: Remove checkpoint file

#### Test: `test_early_termination_pruning`

**File**: `tests/integration/hpo/test_early_termination.py`

- **Setup**:
- Mock trials with decreasing metrics
- Early termination: bandit, evaluation_interval=1, slack_factor=0.2, delay_evaluation=2
- **Execution**: Run HPO with max_trials=5
- **Assertions**:
- First 2 trials not pruned (delay_evaluation=2)
- Trial 3+ pruned if metric < (best - slack_factor * best)
- Pruned trials marked PRUNED in Optuna
- Pruned trials have code.interrupted tag (if cleanup enabled)
- **Teardown**: None

#### Test: `test_refit_training`

**File**: `tests/integration/hpo/test_refit_training.py`

- **Setup**:
- Complete HPO with best trial selected
- refit.enabled=true
- **Execution**: Call `run_refit_training()`
- **Assertions**:
- Refit run created as child of best trial run
- Refit run name: `{env}_{model}_hpo_refit_study-{hash}_trial-{hash}_t{trial_number}`
- Refit output: `trial-{hash}/refit/checkpoint/`
- Refit trained on full dataset (no CV splits)
- Refit tags: `code.refit=true`, `code.trained_on_full_data=true`
- Refit run remains RUNNING until artifacts logged
- **Teardown**: Clean up MLflow runs

#### Test: `test_best_trial_selection`

**File**: `tests/integration/hpo/test_best_trial_selection.py`

- **Setup**:
- Mock HPO with 3 trials, different metrics and inference times
- selection.accuracy_threshold=0.015, use_relative_threshold=true
- **Execution**: Call best trial selection logic
- **Assertions**:
- Best trial selected based on objective (macro-f1 maximize)
- Speed tradeoff applied when within threshold
- Best trial tags logged: `code.hpo.best_trial_run_id`, `code.hpo.best_trial_number`
- Best trial checkpoint logged to MLflow if log_best_checkpoint=true
- Cache file created: `outputs/cache/best_configurations/best_config_{backbone}_{trial}_{timestamp}.json`
- Latest pointer updated: `latest_best_configuration.json`
- Index updated: `index.json`
- **Teardown**: Remove cache files

#### Test: `test_checkpoint_save_only_best`

**File**: `tests/integration/hpo/test_checkpoint_cleanup.py`

- **Setup**:
- Run HPO with 3 trials, checkpoint.save_only_best=true
- Mock training to create checkpoints
- **Execution**: Complete HPO, cleanup runs
- **Assertions**:
- Only best trial checkpoint directory exists
- Other trial checkpoints removed
- Fold checkpoints removed (if CV)
- Refit checkpoint kept (if refit enabled)
- **Teardown**: Remove output directory

#### Test: `test_cleanup_auto_cleanup_enabled`

**File**: `tests/integration/hpo/test_cleanup_behavior.py`

- **Setup**:
- Previous HPO run with interrupted runs (simulated)
- cleanup.disable_auto_cleanup=false (cleanup enabled)
- **Execution**: 
- Run HPO with cleanup enabled
- Call cleanup functions
- **Assertions**:
- Interrupted runs tagged with `code.interrupted=true`
- Cleanup runs automatically on HPO startup
- No cleanup skipped (disable_auto_cleanup=false means enabled)
- Interrupted runs identified correctly
- **Teardown**: Clean up test runs

#### Test: `test_cleanup_optuna_mark_enabled`

**File**: `tests/integration/hpo/test_cleanup_behavior.py`

- **Setup**:
- Previous HPO run with RUNNING trials in Optuna (simulated)
- cleanup.disable_auto_optuna_mark=false (marking enabled)
- Checkpoint file exists
- **Execution**: 
- Resume HPO with cleanup enabled
- Optuna marking runs
- **Assertions**:
- RUNNING trials marked as FAILED in Optuna
- Optuna state updated correctly
- Marking runs automatically (disable_auto_optuna_mark=false means enabled)
- Study state consistent after marking
- **Teardown**: Remove checkpoint file

#### Test: `test_mlflow_structure`

**File**: `tests/integration/hpo/test_mlflow_structure.py`

- **Setup**: Run full HPO with CV and refit
- **Execution**: Query MLflow runs via client
- **Assertions**:
- Parent run: `code.stage=hpo`, `code.run_type=hpo_sweep`
- Trial runs: `mlflow.parentRunId=parent_id`, `code.run_type=hpo_trial`
- Fold runs: `mlflow.parentRunId=trial_id`, `code.run_type=hpo_trial_fold`
- Refit run: `mlflow.parentRunId=best_trial_id`, `code.run_type=hpo_refit`
- All runs have: `code.study_key_hash`, `code.trial_key_hash` (where applicable)
- Grouping tags consistent across related runs
- **Teardown**: Clean up MLflow runs

#### Test: `test_path_structure_v2`

**File**: `tests/integration/hpo/test_path_structure.py`

- **Setup**: Run HPO with CV and refit
- **Execution**: Check filesystem structure
- **Assertions**:
- Study folder: `outputs/hpo/{storage_env}/{model}/study-{study8}/`
- Trial folders: `study-{study8}/trial-{trial8}/`
- CV structure: `trial-{trial8}/cv/fold0/checkpoint/`, `cv/fold1/checkpoint/`
- Refit structure: `trial-{trial8}/refit/checkpoint/`
- trial_meta.json in each trial folder
- fold_splits.json in study folder
- **Teardown**: Remove output directory

#### Test: `test_environment_path_overrides`

**File**: `tests/integration/hpo/test_environment_paths.py`

- **Setup**: Mock platform detection for colab, kaggle, azureml
- **Execution**: Run HPO in each environment
- **Assertions**:
- Colab: outputs = `/content/drive/MyDrive/resume-ner-azureml/outputs`
- Kaggle: outputs = `/kaggle/working/outputs`
- AzureML: outputs = `/mnt/outputs`
- Local: outputs = `{project_root}/outputs`
- Paths resolved correctly via `resolve_output_path()`
- **Teardown**: None

### 4.3 Integration Tests

#### Test: `test_full_hpo_workflow_smoke`

**File**: `tests/e2e/test_hpo_full_workflow.py`

- **Setup**:
- Load smoke.yaml config
- Tiny dataset (10 samples)
- Mock training (fast, returns metrics)
- Mock MLflow (local SQLite)
- **Execution**:
- Load configs (data, model, train, hpo, env, benchmark)
- Setup MLflow experiment and parent run
- Create Optuna study
- Run 1 trial with 2-fold CV
- Select best trial
- Run refit training
- Log best checkpoint to MLflow
- **Assertions**:
- All configs loaded correctly
- Study created with correct name
- 1 trial completed
- 2 fold runs created
- CV metrics aggregated
- Best trial selected
- Refit run created and completed
- Best checkpoint logged to MLflow
- Cache files created (best_config, latest, index)
- All paths follow v2 structure
- All MLflow tags present and correct
- **Teardown**: Remove all test outputs

#### Test: `test_smoke_yaml_all_settings_applied`

**File**: `tests/e2e/test_smoke_yaml_complete.py`

- **Setup**:
- Load exact smoke.yaml (no modifications)
- Tiny dataset
- Mock training
- Mock MLflow
- **Execution**:
- Run complete HPO workflow with smoke.yaml
- **Assertions**:
- **Search space**: backbone=distilbert, learning_rate in [1e-5, 5e-5], batch_size=4, dropout in [0.1, 0.3], weight_decay in [0.001, 0.1]
- **Sampling**: algorithm=random, max_trials=1, timeout_minutes=20
- **Checkpoint**: enabled=true, study_name="hpo_distilbert_smoke_test_path_testing_23", storage_path resolves correctly, auto_resume=true, save_only_best=true
- **MLflow**: log_best_checkpoint=true
- **Early termination**: policy=bandit, evaluation_interval=1, slack_factor=0.2, delay_evaluation=2
- **Objective**: metric=macro-f1, goal=maximize
- **Selection**: accuracy_threshold=0.015, use_relative_threshold=true, min_accuracy_gain=0.02
- **k_fold**: enabled=true, n_splits=2, random_seed=42, shuffle=true, stratified=true
- **Refit**: enabled=true
- **Cleanup**: disable_auto_cleanup=false (cleanup enabled), disable_auto_optuna_mark=false (marking enabled)
- All settings validated simultaneously in single run
- **Teardown**: Remove all test outputs

#### Test: `test_hpo_resume_workflow`

**File**: `tests/e2e/test_hpo_resume_workflow.py`

- **Setup**: Same as above, but interrupt after trial 0
- **Execution**:
- Resume HPO
- Complete remaining trials
- **Assertions**:
- Study resumes (not new study)
- No duplicate trials
- All trials complete
- Best trial selected from all trials
- **Teardown**: Remove all test outputs

#### Test: `test_hpo_with_cleanup`

**File**: `tests/e2e/test_hpo_cleanup.py`

- **Setup**: Previous interrupted HPO run (simulated)
- **Execution**:
- Run HPO with cleanup.enabled=true
- Check interrupted runs
- **Assertions**:
- Interrupted runs tagged with `code.interrupted=true`
- RUNNING Optuna trials marked as FAILED
- New HPO run proceeds normally
- **Teardown**: Clean up

## 5. Mocking Strategy

### What to Mock

#### MLflow Client

- **When**: All tests (unit, component, integration)
- **How**: Use `unittest.mock.patch` or `pytest-mock` to mock `mlflow.tracking.MlflowClient()`
- **Mock Returns**:
- `create_run()` returns mock run with `info.run_id`
- `get_run()` returns mock run with tags, metrics, params
- `log_metric()`, `log_param()`, `log_artifact()` return None
- `set_terminated()` returns None
- **Alternative**: Use local SQLite MLflow backend for integration tests (real but isolated)

#### Azure ML Services

- **When**: Tests that involve Azure ML (azureml environment)
- **How**: Mock `azure.ai.ml.MLClient` and related classes
- **Mock Returns**: No-op for all methods (environments, jobs, etc.)

#### Google Drive Sync

- **When**: Colab environment tests
- **How**: Mock Drive mount check and file operations
- **Mock Returns**: Simulate Drive paths, no actual Drive access

#### Training Subprocess

- **When**: Component and integration tests (to avoid real training)
- **How**: Mock `subprocess.run()` or training entry point
- **Mock Returns**:
- Return code 0 (success)
- Create mock `metrics.json` file
- Create mock checkpoint directory structure
- **Note**: Unit tests for training logic should test training code directly (not mocked)

#### Optuna Study (selective)

- **When**: Unit tests for selection logic, path building
- **How**: Create mock Optuna trial objects with attributes
- **Mock Returns**: Trial with `number`, `params`, `user_attrs`, `state`
- **Note**: Integration tests use real Optuna study (with mocked training)

### What to Run Real

#### Filesystem Operations

- **When**: All tests
- **Why**: Need to validate actual path creation, file writing, directory structure
- **How**: Use `tmp_path` fixture (pytest) or `tempfile.mkdtemp()`

#### Config Loading

- **When**: All tests
- **Why**: Need to validate YAML parsing, config resolution
- **How**: Use real YAML files in test fixtures

#### Path Resolution

- **When**: All tests
- **Why**: Need to validate path building logic
- **How**: Use real `resolve_output_path()`, `build_output_path()` functions

#### Hash Computation

- **When**: All tests
- **Why**: Need to validate deterministic hashing
- **How**: Use real hash functions

#### Optuna Study (integration)

- **When**: Integration tests
- **Why**: Need to validate study creation, resume, pruning
- **How**: Use real Optuna with SQLite storage backend

## 6. Fixtures and Test Utilities

### Fixtures (conftest.py)

#### `tmp_config_dir`

- Creates temporary config directory with all required YAML files
- Includes: smoke.yaml, paths.yaml, naming.yaml, tags.yaml, mlflow.yaml, data.yaml, model.yaml, train.yaml, env.yaml, benchmark.yaml
- Returns: Path to config directory

#### `tiny_dataset`

- Creates minimal dataset (10 samples) in JSONL format
- Returns: Path to dataset directory

#### `mock_mlflow_client`

- Provides mocked MLflow client with common operations
- Returns: Mock client instance

#### `mock_training_subprocess`

- Mocks training subprocess to return success and create metrics.json
- Returns: Mock subprocess function

#### `hpo_config_smoke`

- Loads and returns smoke.yaml HPO config
- Returns: Dict with HPO configuration

#### `tmp_output_dir`

- Creates temporary output directory
- Returns: Path to output directory

### Test Utilities

#### `create_mock_trial(trial_number, params, metric_value)`

- Creates mock Optuna trial object
- Returns: Mock trial with attributes

#### `create_mock_mlflow_run(run_id, tags, metrics, params)`

- Creates mock MLflow run object
- Returns: Mock run with data

#### `assert_path_structure_v2(output_dir, expected_structure)`

- Validates v2 path structure
- Checks: study-{hash}, trial-{hash}, cv/foldX, refit folders

#### `assert_mlflow_tags(run, expected_tags)`

- Validates MLflow tags present and correct
- Checks: grouping tags, process tags, lineage tags

#### `assert_cache_files(cache_dir, cache_type, expected_entries)`

- Validates cache file structure
- Checks: timestamped file, latest pointer, index file

#### `load_golden_outputs(test_name)`

- Loads expected outputs for comparison
- Returns: Dict with expected paths, tags, metrics

## 7. Failure/Negative Tests

### Test: `test_missing_config_keys`

**File**: `tests/integration/hpo/test_error_handling.py`

- **Setup**: Config with missing required keys
- **Execution**: Try to load and use config
- **Assertions**: Appropriate defaults used or clear error message

### Test: `test_invalid_hyperparameter_range`

**File**: `tests/unit/orchestration/test_search_space.py`

- **Setup**: Search space with min > max
- **Execution**: Try to translate to Optuna
- **Assertions**: ValidationError raised

### Test: `test_corrupted_checkpoint_file`

**File**: `tests/integration/hpo/test_error_handling.py`

- **Setup**: Corrupted study.db file
- **Execution**: Try to resume study
- **Assertions**: Error handled, new study created or clear error

### Test: `test_mlflow_tracking_failure`

**File**: `tests/integration/hpo/test_error_handling.py`

- **Setup**: MLflow client raises exception
- **Execution**: Run HPO
- **Assertions**: HPO continues (MLflow optional), errors logged

### Test: `test_trial_training_failure`

**File**: `tests/integration/hpo/test_error_handling.py`

- **Setup**: Training subprocess returns error
- **Execution**: Run trial
- **Assertions**: Trial marked FAILED, study continues with next trial

### Test: `test_path_creation_failure`

**File**: `tests/integration/hpo/test_error_handling.py`

- **Setup**: Output directory not writable
- **Execution**: Try to create paths
- **Assertions**: Clear error message, no partial writes

### Test: `test_invalid_hash_computation`

**File**: `tests/unit/orchestration/test_mlflow_naming.py`

- **Setup**: Missing required config for hash
- **Execution**: Try to compute study_key_hash
- **Assertions**: Error handled gracefully, hash set to None or default

## 8. Acceptance Criteria

### Must Pass (Blocking)

1. All scenarios in smoke.yaml have corresponding tests
2. All unit tests pass (pure functions, deterministic)
3. All component tests pass (orchestrator functions with mocks)
4. Full HPO workflow integration test passes (end-to-end)
5. Resume/restart behavior validated (no duplicates, correct state)
6. MLflow structure validated (parent/child relationships, tags)
7. Path structure validated (v2 patterns, environment overrides)
8. Cache files validated (dual file strategy, latest pointer, index)
9. Best trial selection validated (criteria, tie-breaking)
10. Error handling validated (graceful failures, no crashes)
11. Template resolution validated (study_name, storage_path placeholders)
12. Cleanup behavior validated (when flags are false = enabled)

### Should Pass (Non-blocking but important)

1. Performance: Full workflow completes in <5 minutes with mocked training
2. Coverage: >90% code coverage for HPO orchestration code
3. Documentation: All test cases documented with clear assertions

### Nice to Have

1. Visualizations: Test outputs include diagrams of MLflow structure
2. Golden files: Expected outputs saved for regression testing
3. Property-based tests: Use hypothesis to test edge cases

## 9. Implementation Checklist

### Phase 1: Unit Tests (Foundation)

- [ ] `tests/unit/orchestration/test_hpo_search_space.py` - Search space translation
- [ ] `tests/unit/orchestration/test_config_loader.py` - Config loading and hashing
- [ ] `tests/unit/orchestration/test_mlflow_naming.py` - Hash computation, run naming
- [ ] `tests/unit/orchestration/test_naming_centralized.py` - Run name generation
- [ ] `tests/unit/orchestration/test_paths.py` - Path building v2
- [ ] `tests/unit/training/test_cv_utils.py` - K-fold split creation
- [ ] `tests/unit/orchestration/test_best_trial_selection.py` - Selection criteria

### Phase 2: Component Tests (Orchestration)

- [ ] `tests/integration/hpo/test_hpo_sweep_setup.py` - Study and MLflow setup
- [ ] `tests/integration/hpo/test_checkpoint_resolution.py` - Template resolution (study_name, storage_path)
- [ ] `tests/integration/hpo/test_trial_execution.py` - Trial execution (no CV, with CV)
- [ ] `tests/integration/hpo/test_hpo_checkpoint_resume.py` - Checkpoint and resume
- [ ] `tests/integration/hpo/test_early_termination.py` - Pruning behavior
- [ ] `tests/integration/hpo/test_refit_training.py` - Refit execution
- [ ] `tests/integration/hpo/test_best_trial_selection.py` - Selection logic
- [ ] `tests/integration/hpo/test_checkpoint_cleanup.py` - Cleanup behavior
- [ ] `tests/integration/hpo/test_cleanup_behavior.py` - Cleanup enabled behavior (auto_cleanup, optuna_mark)
- [ ] `tests/integration/hpo/test_mlflow_structure.py` - MLflow validation
- [ ] `tests/integration/hpo/test_path_structure.py` - Path validation
- [ ] `tests/integration/hpo/test_environment_paths.py` - Environment overrides
- [ ] `tests/integration/hpo/test_error_handling.py` - Failure cases

### Phase 3: Integration Tests (End-to-End)

- [ ] `tests/e2e/test_hpo_full_workflow.py` - Complete smoke.yaml workflow
- [ ] `tests/e2e/test_smoke_yaml_complete.py` - All smoke.yaml settings applied together
- [ ] `tests/e2e/test_hpo_resume_workflow.py` - Resume scenario
- [ ] `tests/e2e/test_hpo_cleanup.py` - Cleanup scenario

### Phase 4: Test Infrastructure

- [ ] `tests/conftest.py` - Shared fixtures (tmp_config_dir, tiny_dataset, mocks)
- [ ] `tests/integration/hpo/conftest.py` - HPO-specific fixtures
- [ ] `tests/helpers/test_assertions.py` - Custom assertion helpers
- [ ] `tests/helpers/test_utils.py` - Test utility functions

### Phase 5: Documentation

- [ ] Update `tests/README.md` with HPO test documentation
- [ ] Add docstrings to all test files
- [ ] Create test execution guide

## 10. Execution Order

1. **Unit tests first** (fast, no dependencies)
2. **Component tests** (require mocks, still fast)
3. **Integration tests** (require full setup, slower)
4. **Run in CI/CD** with pytest, generate coverage report

## 11. Test Data Requirements

- **Tiny dataset**: 10-20 samples (train.jsonl, val.jsonl)
- **Config files**: All YAML files from config/ directory (as fixtures)
- **Golden outputs**: Expected MLflow structure, paths, cache files (for regression)

## 12. Notes

- Tests should be deterministic (fixed seeds, no randomness in assertions)
- Use pytest markers to categorize tests (`@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`)
- Tests should be isolated (no shared state between tests)
- Use `tmp_path` for all file operations (automatic cleanup)
- Mock external services but test real orchestration logic
- Focus on contracts (what outputs are created, not implementation details)

### To-dos

- [ ] Implement unit tests for search space translation (test_hpo_search_space.py)
- [ ] Implement unit tests for config loading and hashing (test_config_loader.py)
- [ ] Implement unit tests for MLflow naming and hash computation (test_mlflow_naming.py, test_naming_centralized.py)
- [ ] Implement unit tests for path building v2 (test_paths.py)
- [ ] Implement unit tests for k-fold split creation (test_cv_utils.py)
- [ ] Implement unit tests for best trial selection criteria (test_best_trial_selection.py)
- [ ] Implement component tests for HPO sweep setup (test_hpo_sweep_setup.py)
- [ ] Implement component tests for trial execution with and without CV (test_trial_execution.py)
- [ ] Implement component tests for checkpoint resume (test_hpo_checkpoint_resume.py)
- [ ] Implement component tests for early termination pruning (test_early_termination.py)
- [ ] Implement component tests for refit training (test_refit_training.py)
- [ ] Implement component tests for best trial selection (test_best_trial_selection.py)
- [ ] Implement component tests for MLflow structure validation (test_mlflow_structure.py)
- [ ] Implement component tests for path structure validation (test_path_structure.py)
- [ ] Implement component tests for error handling (test_error_handling.py)
- [ ] Implement integration test for full HPO workflow (test_hpo_full_workflow.py)
- [ ] Implement integration test for resume workflow (test_hpo_resume_workflow.py)
- [ ] Create shared test fixtures and utilities (conftest.py, test_assertions.py, test_utils.py)
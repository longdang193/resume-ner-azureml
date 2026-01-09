<!-- 146ac483-fce0-437a-95e2-2625602b4b4b 2387fdab-3317-4563-bce3-c0a46ebe9ce0 -->
# Training Process Test Plan (Updated)

## 1. Scope and assumptions

- Cover final training orchestration driven by `notebooks/02_best_config_selection.ipynb`, configs: `config/final_training.yaml`, `config/train.yaml`, `config/paths.yaml`, `config/naming.yaml`, `config/tags.yaml`, `config/mlflow.yaml`, experiment/data/model/HPO configs, env overrides (`config/env/*`).
- Environments/storage_env: local, colab, kaggle, azureml; external services (MLflow/Azure) mocked unless noted; filesystem real (tmp dirs).
- Notebook order is authoritative (env detect → setup → config load → MLflow setup → drive backup → best model selection + cache → skip/retrain decision → final training exec → drive backup → conversion). Conversion tested as handoff validation (not ONNX correctness).
- No scenarios beyond YAML/notebook.

## 2. Inventory of scenarios derived from YAML

| yaml case | notebook step(s) | scripts/modules | test level | expected outputs |
|---|---|---|---|---|
| run.mode: reuse_if_exists, force_new, resume_if_incomplete, continue_from_previous | Step 7 | `orchestration.jobs.final_training.execute_final_training`, cache helpers | component/integration | Variant decision correct; reuse/skip vs new run; cache updated/indexed |
| source.type: scratch, best_selected, final_training (parent null/spec/exec/variant) | Steps 6–7 | `extract_lineage_from_best_model`, `acquire_best_model_checkpoint`, checkpoint loader | component | Checkpoint resolution per type; lineage tags set; warnings on conflicts |
| checkpoint.load/validate | Step 7 | `training.checkpoint_loader` | unit/component | Enforces validation when true; warnings/failures when missing |
| dataset.data_config override + local_path_override | Step 3/7 | config loader, dataset resolver | component | Override respected; error when missing |
| variant.number with run.mode interactions | Step 7 | naming/path builders | component | Auto-increment unless force_new ignores explicit |
| identity flags include_code_fp/precision_fp/determinism_fp | Step 7 | fingerprint computation | unit | exec/spec FP reflect flags |
| seed.random_seed override | Step 7 | training config merge | unit | Seed propagated |
| Training overrides (lr/epochs/batch/dropout/weight_decay/grad_acc/warmup/max_grad_norm/early_stopping.*) | Step 7 | config merger | unit/component | Null keeps defaults; overrides applied |
| MLflow overrides + tracking toggles (`mlflow.yaml`) | Steps 4/7 | `setup_mlflow_from_config`, run-name builder | component | Names/tags match overrides; disabled tracking yields no runs |
| Paths env_overrides + storage_env | Steps 1–2 | `orchestration.paths`, drive backup | integration | Output roots per env; drive backup only colab |
| Cache strategies (`cache_strategies.*`) | Steps 6–7 | cache helpers | component/integration | dual-file latest/index/timestamped maintained |
| HPO/selection configs (`hpo/*.yaml`, `best_model_selection.yaml`) | Step 6 | selection logic | unit/component | Thresholds/tie-breaks per config |
| Model configs (`model/*.yaml`), data variants (`data/*.yaml`) | Steps 3/7 | trainer setup | component | Sequence length/CRF flags honored |
| Environment configs (`env/*.yaml`) | Steps 1–4 | MLflow/AML setup | component | AML enable/disable respected |

## 3. Existing test audit (`tests`)

- Unit: training (`tests/unit/training/*`) covers data loaders, checkpoints, k-fold, dataset combiner; orchestration/naming (`tests/unit/orchestration/*`) covers naming/paths/fingerprints; selection logic (`test_best_trial_selection.py`).
- Integration: `tests/integration/orchestration/test_config_integration.py` for naming/paths; `tests/integration/hpo/test_best_trial_selection_component.py` for Optuna selection.
- E2E: `tests/e2e/test_e2e_workflow.py` tiny local HPO→selection→optional training.
- Gaps to fill: final_training orchestration (run modes, resume/continue), cache dual-file atomicity, MLflow enable/disable, platform storage_env paths, resume state fidelity, completion contract, data invariants, conversion handoff. Add appendix mapping existing files→contracts to avoid duplication.

## 4. Test matrix

| scenario | environment | storage_env | tracking | resume mode | expected outputs |
|---|---|---|---|---|---|
| Final training best_selected | local | local | on | force_new | outputs/final_training/local/{model}/spec-{spec8}_exec-{exec8}/v1 with checkpoint/, metrics.json, metadata.json, tokenizer/config, cache latest/index |
| reuse_if_exists | local | local | on | reuse_if_exists | Existing complete run detected; no new variant; cache unchanged |
| resume incomplete | local | local | on | resume_if_incomplete | Same variant resumes; trainer state restored; logs aligned |
| continue from previous | local | local | on | continue_from_previous | Uses parent spec/exec/variant; lineage tags set |
| source scratch | local | local | on | any | No checkpoint load |
| explicit checkpoint path | local | local | on | force_new | Validated path; errors if missing |
| dataset override | local | local | on | force_new | Override dataset used; missing raises |
| MLflow disabled | local | local | off | force_new | No runs; FS outputs + cache still written |
| Colab + Drive | colab | colab | on | force_new | Env override paths; drive backup called |
| Kaggle | kaggle | kaggle | on | force_new | Kaggle paths; no drive backup |
| AzureML | azureml | azureml | on | force_new | AML experiment naming; workspace URI used |
| Auto-increment run names | any | any | on | force_new | Names increment per `mlflow.yaml` |
| Identity flags toggle | any | any | on | force_new | FP changes per flags |
| Selection cache reuse | any | any | on | n/a | best_model loaded from cache vs recompute |
| Conversion handoff | any | any | on | post-train | parent_training_id parsed; conversion blocked if artifacts missing |

## 5. Detailed test cases (additions emphasized)

- **Unit (utilities)**
- Config merge for final_training vs train defaults; null-handling; seed propagation.
- Fingerprint/identity toggles affect exec/spec FP.
- Variant resolution per run.mode.
- Naming/tag building (final_training) respects storage_env, variant, spec/exec; shorten flag.
- MLflow enable/disable paths in `setup_mlflow_from_config` (azure_ml.enabled on/off, tracking.training.enabled false).
- Selection logic: thresholds/tie-breaks, missing metrics/NaN fallback, scoring weights from `best_model_selection.yaml`.
- **Resume state fidelity**: if resume restores optimizer/scheduler/scaler/global step/epoch/ES state; if design is weights-only, assert that explicitly.
- **Completion contract**: function deciding “complete” checks required files and non-empty/valid JSON; corrupt/empty files mark incomplete.
- **Data invariants**: schema validation, empty dataset handling, tokenization max length truncation flags, label map consistency.

- **Component (orchestration)**
- Best model selection cache load/save: force_new vs reuse_if_exists; dual-file correctness.
- Final training cache strategy: timestamped/latest/index updated atomically; simulate partial write (latest missing, index updated) → recovery.
- Checkpoint acquisition for source types; validate flag behavior.
- Training context builder: storage_env per env_overrides; path/name/tag coherence (colab/kaggle/azureml/local).
- Resume/continue modes: existing outputs with metadata indicating complete/incomplete; race on variant/auto-increment (simulate concurrent starts) to ensure no duplicate variant.
- MLflow logging: mock client; tags, params, artifacts (including tokenizer/config/label map); run name auto-increment; disabled tracking path.
- Dataset overrides; identity flags; seed propagation into trainer call.
- **Metric selection contract**: best checkpoint metric key, tie-break rule (timestamp/step), fallback when missing/NaN; alignment with notebook selection.
- **Path edge cases**: local_path_override with Windows separators/backslashes, UNC/drive letters, path length normalization/limit.

- **Integration (notebook-mirroring)**
- Local happy path: mock MLflow + trainer writing checkpoint/, metrics.json, metadata.json (with spec_fp/exec_fp/variant/run_id), tokenizer/config/label map, logs; cache updated; conversion invoked with parent_training_id parsed per naming.
- Colab path + drive backup; Kaggle path; AzureML tracking enabled (mock workspace) using experiment naming from experiment config.
- Tracking disabled path: no MLflow runs but FS outputs/caches produced.
- Resume notebook rerun: second invocation with existing cache/outputs; run.mode=reuse_if_exists skips work; resume_if_incomplete continues same variant with restored trainer state.
- Error paths: missing dataset; invalid checkpoint; corrupted cache (latest pointing missing) → recompute; MLflow experiment missing → fallback/local URI; conversion fails fast when required artifacts absent.
- **Data edge cases**: empty train/eval splits, schema mismatch/missing labels, tiny dataset no batches.

## 6. Mocking strategy

- MLflow client/start_run; Azure ML SDK; drive backup.
- Trainer stub writes checkpoint/, metrics.json, metadata.json, tokenizer/config/label map, logs.
- Time/uuid freeze for deterministic cache filenames and run names.
- Concurrency: simulate dual calls to variant/auto-increment functions.

## 7. Fixtures and utilities

- Config-builder fixture cloning YAMLs with override hooks.
- Fake MLflow run registry fixture producing benchmark/HPO/final runs with required tags/metrics, including missing/NaN cases.
- Stub trainer fixture that can simulate resume restoring states or weights-only behavior.
- Cache seeding fixture for latest/index and partial-write scenarios.
- Platform fixture to set COLAB/KAGGLE flags and storage_env mapping; Windows path normalization helper.

## 8. Negative/failure tests

- Invalid YAML keys/values (e.g., variant non-int, unsupported source.type).
- Missing dataset files/override path; empty datasets; schema mismatch.
- Checkpoint load true but path missing/invalid; validate false bypass.
- Cache corruption (latest/index mismatch); partial write; selection cache missing files.
- MLflow failures (experiment not found, URI unset); Azure credentials missing (graceful failure/mocking).
- Resume_if_incomplete with no partial outputs → new run; incomplete metadata lacking metrics; best checkpoint missing while metrics exist.
- Conversion preconditions missing metadata.spec_fp/exec_fp or artifacts → clear error.

## 9. Acceptance criteria

- Every YAML/notebook scenario and added gotchas (resume state, completion contract, cache atomicity, data invariants, metric selection) has a single owning test layer.
- Notebook flow mirrored; rerun/resume behavior verified; no duplicate coverage.
- Filesystem layout per `paths.yaml`; MLflow tags/names per configs; cache dual-file integrity under normal and partial writes.
- Negative paths raise clear, expected errors.

## 10. Implementation checklist

- Append an audit appendix mapping existing tests → covered contracts; mark ownership to avoid duplication.
- Unit: add under `tests/unit/orchestration` (config merge, identity, naming, completion checker, metric selection fallback), `tests/unit/training` (resume state fidelity, data invariants where unit-scoped).
- Component: add `tests/integration/orchestration/test_final_training_component.py` (run modes, cache atomicity, MLflow on/off, paths, resume state), `tests/integration/selection/test_best_model_cache.py` (cache dual write, missing/NaN metrics), Windows path edge cases.
- Integration: add `tests/e2e/test_notebook_training_flow.py` (env matrix local/colab/kaggle, tracking on/off, resume rerun, data edge cases) plus conversion handoff checks; AzureML mocked variant in same file.
- Order: (1) audit appendix, (2) unit config/naming/completion/metrics, (3) unit resume-state semantics, (4) component final_training/cache/path edge cases, (5) integration env/resume matrix + conversion handoff, (6) negative/error paths.

Appendix (existing closest tests)

- Naming/paths/tags: `tests/unit/orchestration/test_naming_*`, `test_path_building_comprehensive.py`, `tests/integration/orchestration/test_config_integration.py`.
- Selection logic: `tests/unit/orchestration/test_best_trial_selection.py`, `tests/integration/hpo/test_best_trial_selection_component.py`.
- Training helpers: `tests/unit/training/test_trainer.py`, `test_checkpoint_loader.py`, `test_cv_utils.py`, `test_data_combiner.py`.
- E2E workflow: `tests/e2e/test_e2e_workflow.py` (local tiny HPO/selection/training) — extend rather than duplicate.

Audit mapping (ownership)

| Contract | Existing coverage | Owner after this plan |
|---|---|---|
| Naming/path patterns | `tests/unit/orchestration/test_naming_*`, `test_path_building_comprehensive.py`, `tests/integration/orchestration/test_config_integration.py` | Keep existing; reuse fixtures |
| HPO selection logic | `tests/unit/orchestration/test_best_trial_selection.py`, `tests/integration/hpo/test_best_trial_selection_component.py` | Extend only for cache/NaN cases |
| Training data loaders/k-fold/combiner | `tests/unit/training/test_trainer.py`, `test_cv_utils.py`, `test_data_combiner.py` | No duplication; add resume/completion/edge cases separately |
| E2E tiny workflow | `tests/e2e/test_e2e_workflow.py` | Extend for resume/rerun matrix instead of rewriting |
| Final training orchestration (run modes, resume state, cache) | none | New unit/component/integration tests |
| Cache dual-file integrity | none | New component tests |
| Conversion handoff validation | none | New integration checks |
| Windows path edge cases | none | New component tests |

### To-dos

- [ ] Summarize YAML scenario matrix
- [ ] Audit existing tests and gaps
- [ ] Design unit cases for config/naming/fingerprint
- [ ] Design component tests for final training/cache
- [ ] Design integration matrix incl. env/resume
- [ ] Define mocking/fixtures strategy
- [ ] List implementation files/order
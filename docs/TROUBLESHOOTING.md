# Troubleshooting Guide

This document provides solutions to common problems encountered when running the Resume NER training pipeline on Azure ML.

## Table of Contents

- [Data Access Issues](#data-access-issues)

- [Memory Issues](#memory-issues)

- [Tokenizer Errors](#tokenizer-errors)

- [Dependency and Compatibility Issues](#dependency-and-compatibility-issues)

- [Job Configuration Errors](#job-configuration-errors)

- [Model Loading Issues](#model-loading-issues)

- [MLflow Issues](#mlflow-issues)

- [FastAPI Service Issues](#fastapi-service-issues)

## Data Access Issues

### Problem: `ScriptExecution.StreamAccess.NotFound` Error

#### Symptoms

- Job fails with error: `ScriptExecution.StreamAccess.NotFound`

- Error occurs when remote jobs try to access data assets

- Particularly seen in data validation jobs (e.g., `data-ls-test`)

#### Root Cause

Using manual datastore path construction instead of Azure ML asset references can cause permission and path resolution issues.

#### Solution

Use Azure ML asset references (`azureml:name:version`) instead of manually constructing datastore paths.

**Before (Incorrect):**

```python
default_datastore = ml_client.datastores.get_default()
relative_path = data_asset.path.split("/paths/", 1)[1]
data_path = f"azureml://datastores/{default_datastore.name}/paths/{relative_path}"
data_input = Input(type="uri_folder", path=data_path)

```

**After (Correct):**

```python
data_input = Input(
    type="uri_folder",
    path=f"azureml:{data_asset.name}:{data_asset.version}",
    mode="mount",
)

```

#### Why This Works

- Azure ML automatically resolves asset references to the correct datastore paths

- The compute cluster's managed identity has proper permissions for asset references

- Asset references are versioned and immutable, ensuring reproducibility

#### Prevention

Always use asset references (`azureml:name:version`) when creating `Input` objects for data assets in sweep jobs and command jobs.

## Memory Issues

### Problem: Out of Memory (OOM) Errors

#### Symptoms

- Job fails with: `Out of memory: Killed process`

- Process uses ~6.9GB+ RAM on 8GB VM

- Error appears in `execution-wrapper.log`: `Out of memory: Killed process`

- Particularly common with large models like DeBERTa-v3-base

#### Root Cause

Large transformer models (e.g., DeBERTa-v3-base with ~184M parameters) combined with large batch sizes exceed available VM memory.

#### Solution 1 Reduce Batch Sizes in HPO Config**

Update `config/hpo/smoke.yaml` and `config/hpo/prod.yaml`:

```yaml
batch_size:
  type: "choice"
  values: [4, 8]  # Reduced from [8, 16] for DeBERTa OOM safety

```

#### Solution 2: Automatic Batch Size Reduction in Training Code

The training code automatically caps batch size for large models:

```python
# In src/training/train.py
if "deberta" in backbone.lower() and batch_size > 8:
    print(f"⚠️  Reducing batch_size from {batch_size} to 8 for DeBERTa to prevent OOM")
    batch_size = 8

```

#### Prevention

- Use smaller batch sizes (4-8) for large models on CPU-only VMs

- Consider using GPU compute for larger models

- Monitor memory usage in job logs

## Tokenizer Errors

### Problem: `return_offsets_mapping is not available when using Python tokenizers`

#### Symptoms

- Error: `NotImplementedError: return_offset_mapping is not available when using Python tokenizers`

- Occurs with DeBERTa models during dataset tokenization

- Training fails in `ResumeNERDataset.__getitem__`

#### Root Cause

Slow (Python) tokenizers don't support `return_offsets_mapping`, but the code always requested it.

#### Solution

Check if tokenizer supports offsets before requesting them:

```python
# In src/training/train.py, ResumeNERDataset.__getitem__
supports_offsets = bool(getattr(self.tokenizer, "is_fast", False))

if supports_offsets:
    encoded = self.tokenizer(
        text,
        truncation=True,
        max_length=self.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    labels = encode_annotations_to_labels(text, annotations, offsets, self.label2id)
else:
    encoded = self.tokenizer(
        text,
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt",
    )
    # Fallback: assign all tokens 'O' label when offsets unavailable
    labels = [self.label2id["O"]] * len(encoded["input_ids"][0])

```

#### Prevention

- Prefer fast tokenizers where available

- Always check tokenizer capabilities before using advanced features

## Dependency and Compatibility Issues

### Problem: NumPy ABI Incompatibility

#### Symptoms

- Error: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6`

- Occurs when loading PyTorch or other compiled modules

#### Solution

Pin NumPy to version < 2.0 in `config/environment/conda.yaml`:

```yaml
dependencies:
  - numpy>=1.24.0,<2.0

```

### Problem: Missing SentencePiece Dependency

#### Symptoms

- Error: `DebertaV2Tokenizer requires the SentencePiece library`

- Occurs when using DeBERTa slow tokenizer

#### Solution

Add SentencePiece to `config/environment/conda.yaml`:

```yaml
dependencies:
  - pip:
      - sentencepiece>=0.1.99

```

### Problem: PyTorch `torch.load` Vulnerability

#### Symptoms

- Error: `Due to a serious vulnerability issue in torch.load, ... upgrade torch to at least v2.6`

- Security check fails when loading model weights

#### Solution

Upgrade PyTorch and use safetensors format:

```yaml
# In config/environment/conda.yaml
dependencies:
  - pytorch>=2.6.0
  - pip:
      - safetensors>=0.4.0

```

```python
# In src/training/train.py
model = AutoModelForTokenClassification.from_pretrained(
    backbone,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    use_safetensors=True,  # Prefer safetensors format
)

```

## Job Configuration Errors

### Problem: Incorrect AML Placeholders

#### Symptoms

- Error: `train.py: error: argument --batch-size: expected one argument`

- Hyperparameters not being injected into command arguments

#### Root Cause

Azure ML placeholders require double braces `${{{{...}}}}` in Python f-strings.

#### Solution

Use double braces for AML placeholders:

```python
command_args = (
    f"--data-asset ${{{{inputs.data}}}} "
    f"--learning-rate ${{{{search_space.learning_rate}}}} "
    f"--batch-size ${{{{search_space.batch_size}}}} "
)

```

**Note:** Single braces `${{...}}` are for YAML, double braces `${{{{...}}}}` are for Python strings.

### Problem: `SweepJob` Parameter Name Error

#### Symptoms

- Error: `ParameterizedSweep.__init__() got an unexpected keyword argument 'early_termination_policy'`

- Occurs when creating HPO sweep jobs

#### Root Cause

`SweepJob` uses `early_termination` parameter, not `early_termination_policy`.

#### Solution

Use correct parameter name:

```python
# Correct
sweep_job = SweepJob(
    trial=trial_job,
    search_space=search_space,
    objective=objective,
    limits=limits,
    early_termination=early_termination,  # ✓ Correct
    compute=compute_cluster,
    inputs={"data": data_input},
)

# Incorrect
sweep_job = SweepJob(
    ...
    early_termination_policy=early_termination,  # ✗ Wrong parameter name
    ...
)

```

### Problem: Config Directory Not Found

#### Symptoms

- Error: `FileNotFoundError: Config directory not found: ../config`
- Error: `FileNotFoundError: Config directory not found: config`

- Occurs in remote jobs

#### Root Cause

Two common misconfigurations cause this:

1. **Code snapshot does not include the `config/` directory** because the
   `code` path is too narrow (e.g. `code="../src"` only snapshots `src/`).
2. **The training command uses the wrong relative path** for `--config-dir`
   (e.g. `--config-dir ../config` even though the working directory already
   contains `config/` at the root of the snapshot).

#### Solution

Set `code=".."` (project root) so both `src/` and `config/` are included, and
use a config path that is relative to that root:

```python
trial_job = command(
    code="..",  # Project root, includes both src/ and config/
    command="python src/training/train.py --config-dir config ...",
    ...
)

```

## Model Loading Issues

### Problem: Hard-Coded Epochs in Dry Run

#### Symptoms

- Dry run sweep ignores epochs from config file

- Always runs with 1 epoch regardless of configuration

#### Root Cause

Hard-coded `--epochs 1` in command arguments.

#### Solution

Remove hard-coded epochs, read from config:

```python
# Remove this:
# f"--epochs 1"

# Epochs will be read from train.yaml config file

```

## MLflow Issues

### Problem: Metrics Not Logged to Azure ML

#### Symptoms

- Training completes successfully but no metrics appear in Azure ML Studio
- `mlflow.log_metric()` calls don't show up in the run
- Selection module can't find metrics for best model selection

#### Root Cause

When running on Azure ML, MLflow requires an active run context. Without calling `mlflow.start_run()`, metrics are logged to a default local context that isn't synced to Azure ML.

#### Solution

Wrap training code with `mlflow.start_run()`:

```python
import mlflow

def main():
    mlflow.start_run()
    
    # Training code here...
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(...)
        mlflow.log_metric("loss", train_loss, step=epoch)
        mlflow.log_metric("macro-f1", f1_score, step=epoch)
    
    mlflow.end_run()
```

#### Additional Requirements

Add `azureml-mlflow` to your environment for Azure ML tracking URI support:

```yaml
# In config/environment/conda.yaml
dependencies:
  - pip:
      - azureml-mlflow>=1.50.0
```

#### Prevention

- Always use `mlflow.start_run()` at the beginning of training scripts
- Use `azureml-mlflow` package when running on Azure ML
- Test metric logging locally before submitting remote jobs

### Problem: Metrics Logged to a Different Run / Unexpected New Job

#### Symptoms

- The Azure ML job you submitted (for example `ivory_planet_*`) shows **no metrics** in
  Azure ML Studio.
- A **separate** MLflow run or Azure ML job appears with metrics (for example
  `tidy_parrot_*`), even though you did not submit it directly.
- Downstream selection or orchestration code that expects metrics on the original job
  fails or cannot find them.

#### Root Cause

When running inside Azure ML, the service **already creates and manages an active MLflow
run** for the job. If your training script unconditionally calls `mlflow.start_run()`,
this creates a **nested child run** (or even a separate job) and all `mlflow.log_*`
calls are attached to that child instead of the top-level job run that Azure ML
tracks in the Studio UI.

#### Solution

- **Detect Azure ML execution and avoid starting a new MLflow run** when one is already
  active. A common pattern:

  ```python
  import mlflow
  import os

  def is_running_in_azure_ml() -> bool:
      # Heuristic: Azure ML sets these env vars for jobs
      return any(
          k in os.environ
          for k in ["AZUREML_RUN_ID", "AZUREML_EXPERIMENT_NAME", "MLFLOW_TRACKING_URI"]
      )

  def main():
      if is_running_in_azure_ml() and mlflow.active_run() is not None:
          # Use the run created by Azure ML; DO NOT call mlflow.start_run() again
          run = mlflow.active_run()
      else:
          # Local or non-Azure context: create a run explicitly
          run = mlflow.start_run()

      # Training code and mlflow.log_* calls here

      if not is_running_in_azure_ml():
          mlflow.end_run()
  ```

- In this pattern:
  - On Azure ML, metrics are logged to the **existing** job run, so they appear on the
    job you submitted.
  - Locally, `mlflow.start_run()` / `mlflow.end_run()` are still used so metrics are
    captured in a local MLflow tracking store.

#### Prevention

- Never unconditionally call `mlflow.start_run()` in a script that will run on Azure ML.
- Always check `mlflow.active_run()` (and optionally Azure ML-specific environment
  variables) before starting a new run.
- Verify in Azure ML Studio that metrics appear on the **same job** you submitted,
  not on a separate child run.

### Problem: Cannot Retrieve Run Metrics for Model Selection

#### Symptoms

- `mlflow.get_run(run_id)` returns empty metrics
- Selection module fails to find best configuration
- Error: `KeyError` when accessing `run.data.metrics`

#### Root Cause

MLflow metrics may not be immediately available after a run completes due to sync delays, or the run context wasn't properly initialized.

#### Solution

Use `mlflow.get_run()` with the correct tracking URI:

```python
import mlflow

# Set tracking URI to Azure ML workspace
mlflow.set_tracking_uri(ml_client.tracking_uri)

# Get run metrics
run = mlflow.get_run(run_id)
metrics = run.data.metrics  # Dict of metric_name -> final_value
```

## General Debugging Tips

### How to Download Job Logs

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from pathlib import Path

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name="resume-ner-ws")
job_id = "your_job_id"

# Download all logs
out_dir = Path("aml_logs") / job_id
out_dir.mkdir(parents=True, exist_ok=True)
ml_client.jobs.download(job_id, all=True, download_path=str(out_dir))

```

### Key Log Files to Check

1. `user_logs/std_log.txt` - Python script output and errors
2. `system_logs/lifecycler/execution-wrapper.log` - System-level errors (OOM, etc.)
3. `azureml-logs/hyperdrive.txt` - Sweep job orchestration logs

### Checking Job Status

```python
job = ml_client.jobs.get(job_id)
print(f"Status: {job.status}")
print(f"Error: {getattr(job, 'error', None)}")

```

### Problem: Checkpoint or ONNX Model Conversion Fails

#### Symptoms

- Conversion job (e.g. `quiet_onion_*`, `purple_wall_*`) fails in `convert_to_onnx.py`
- Error: `ScriptExecution.StreamAccess.NotFound` on a checkpoint input data asset such as
  `azureml_<job>_input_data_checkpoint:1`
- Error: `FileNotFoundError: Checkpoint file not found in: <checkpoint_path>`
- Error: `ModuleNotFoundError: No module named 'onnxscript'`
- Job remains in `Running` state for a long time with no logs, then fails during ONNX export

#### Root Causes

1. **Empty or non-materialized checkpoint output**
   - Training writes no files to the `checkpoint` output folder, so the
     auto-registered data asset (e.g. `azureml_<job>_output_data_checkpoint:1`) is empty.
   - The conversion job uses that empty data asset as its checkpoint input.
2. **Mismatch between training output layout and conversion script**
   - Training saves Hugging Face weights under a nested `checkpoint/` directory
     (e.g. `<AZURE_ML_OUTPUT_DIR>/checkpoint/…`), but the conversion script only
     searched the root folder.
3. **Missing ONNX / onnxscript dependencies**
   - The environment does not include `onnxscript`, so `torch.onnx.export` fails on import.
4. **Slow or hanging ONNX export with no diagnostics**
   - `torch.onnx.export` can be slow or appear to hang on CPU-only compute without
     explicit logging or conservative settings.

#### Solutions

1. **Ensure checkpoint output is always materialized**

   In `src/training/train.py`, always create the `checkpoint` output directory and write a
   placeholder file so Azure ML materializes the output and auto-registers it as a
   data asset:

   ```python
   output_dir = Path(os.getenv("AZURE_ML_OUTPUT_checkpoint") or os.getenv("AZURE_ML_OUTPUT_DIR", "./outputs"))
   output_dir.mkdir(parents=True, exist_ok=True)
   (output_dir / "checkpoint_placeholder.txt").write_text(
       "Placeholder to ensure checkpoint output is materialized."
   )
   ```

2. **Use the auto-registered checkpoint data asset in orchestration**

   In `src/orchestration/jobs/conversion.py`, use
   `get_checkpoint_output_from_training_job` together with `MLClient` to resolve the
   checkpoint input to a proper asset reference like:

   ```python
   checkpoint_input = Input(
       type="uri_folder",
       path=f"azureml:{data_asset.name}:{data_asset.version}",
       mode="mount",
   )
   ```

   This avoids manually constructing datastore URIs and prevents
   `ScriptExecution.StreamAccess.NotFound` on empty or incorrect paths.

3. **Make the conversion script search nested `checkpoint/` directories**

   In `src/model_conversion/convert_to_onnx.py`, search both the root of the mounted checkpoint folder and
   any nested `checkpoint/` subdirectory for standard Hugging Face weight filenames (for
   example `pytorch_model.bin` or `model.safetensors`) before raising `FileNotFoundError`.

4. **Install required ONNX dependencies**

   Add compatible `onnx` and `onnxscript` versions to `config/environment/conda.yaml`:

   ```yaml
   dependencies:
     - numpy>=1.24.0,<2.0
     - onnx>=1.16.0
     - pip:
         - onnxscript>=0.1.0
   ```

5. **Improve ONNX export robustness and logging**

   In `convert_to_onnx`, call `torch.onnx.export` with explicit settings and
   detailed logging, for example:

   ```python
   torch.onnx.export(
       model,
       example_inputs,
       str(onnx_path),
       input_names=["input_ids", "attention_mask"],
       output_names=["logits"],
       dynamic_axes={
           "input_ids": {0: "batch", 1: "seq"},
           "attention_mask": {0: "batch", 1: "seq"},
           "logits": {0: "batch", 1: "seq"},
       },
       opset_version=18,
       do_constant_folding=True,
       dynamo=False,
   )
   ```

   Wrap export, quantization, and smoke tests in `try/except` blocks and log each
   phase with `print(..., flush=True)` or a helper like `_log("Starting FP32 ONNX export…")`
   so you can see long-running steps in `user_logs/std_log.txt`.

#### Prevention

- Keep the training output layout (`<output_dir>/checkpoint/`) and the conversion
  script’s search logic in sync whenever you change where checkpoints are saved.
- Always materialize the `checkpoint` output (even in smoke tests or dry runs) so the
  conversion job’s input data asset is not empty.
- Use asset references (`azureml:name:version`) for checkpoint inputs instead of
  hand-built datastore URIs.
- Pin compatible versions of `numpy`, `onnx`, and `onnxscript` in the environment.
- Use detailed, flushed logging in `convert_to_onnx.py` so you can quickly see
  where a conversion job is stuck.

## Prevention Checklist

Before submitting jobs, verify:

- [ ] Data assets use `azureml:name:version` references

- [ ] Batch sizes are appropriate for model size and VM memory

- [ ] All dependencies are pinned in `conda.yaml`

- [ ] AML placeholders use double braces `${{{{...}}}}`

- [ ] `SweepJob` uses correct parameter names

- [ ] Code snapshot includes all necessary directories (`code=".."`)

- [ ] Training script calls `mlflow.start_run()` for metric logging

## Related Documentation

- [Clean Code Principles](../docs/rules/CLEAN_CODE.md)

- [Documentation Guidelines](../docs/rules/CLEAN_DOC.md)

- [Training Script](../src/training/train.py)

- [Orchestration Notebook](../notebooks/01_orchestrate_training.ipynb)


## General Debugging Tips

### How to Download Job Logs

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from pathlib import Path

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name="resume-ner-ws")
job_id = "your_job_id"

# Download all logs
out_dir = Path("aml_logs") / job_id
out_dir.mkdir(parents=True, exist_ok=True)
ml_client.jobs.download(job_id, all=True, download_path=str(out_dir))

```

### Key Log Files to Check

1. `user_logs/std_log.txt` - Python script output and errors
2. `system_logs/lifecycler/execution-wrapper.log` - System-level errors (OOM, etc.)
3. `azureml-logs/hyperdrive.txt` - Sweep job orchestration logs

### Checking Job Status

```python
job = ml_client.jobs.get(job_id)
print(f"Status: {job.status}")
print(f"Error: {getattr(job, 'error', None)}")

```

### Problem: Checkpoint or ONNX Model Conversion Fails

#### Symptoms

- Conversion job (e.g. `quiet_onion_*`, `purple_wall_*`) fails in `convert_to_onnx.py`
- Error: `ScriptExecution.StreamAccess.NotFound` on a checkpoint input data asset such as
  `azureml_<job>_input_data_checkpoint:1`
- Error: `FileNotFoundError: Checkpoint file not found in: <checkpoint_path>`
- Error: `ModuleNotFoundError: No module named 'onnxscript'`
- Job remains in `Running` state for a long time with no logs, then fails during ONNX export

#### Root Causes

1. **Empty or non-materialized checkpoint output**
   - Training writes no files to the `checkpoint` output folder, so the
     auto-registered data asset (e.g. `azureml_<job>_output_data_checkpoint:1`) is empty.
   - The conversion job uses that empty data asset as its checkpoint input.
2. **Mismatch between training output layout and conversion script**
   - Training saves Hugging Face weights under a nested `checkpoint/` directory
     (e.g. `<AZURE_ML_OUTPUT_DIR>/checkpoint/…`), but the conversion script only
     searched the root folder.
3. **Missing ONNX / onnxscript dependencies**
   - The environment does not include `onnxscript`, so `torch.onnx.export` fails on import.
4. **Slow or hanging ONNX export with no diagnostics**
   - `torch.onnx.export` can be slow or appear to hang on CPU-only compute without
     explicit logging or conservative settings.

#### Solutions

1. **Ensure checkpoint output is always materialized**

   In `src/training/train.py`, always create the `checkpoint` output directory and write a
   placeholder file so Azure ML materializes the output and auto-registers it as a
   data asset:

   ```python
   output_dir = Path(os.getenv("AZURE_ML_OUTPUT_checkpoint") or os.getenv("AZURE_ML_OUTPUT_DIR", "./outputs"))
   output_dir.mkdir(parents=True, exist_ok=True)
   (output_dir / "checkpoint_placeholder.txt").write_text(
       "Placeholder to ensure checkpoint output is materialized."
   )
   ```

2. **Use the auto-registered checkpoint data asset in orchestration**

   In `src/orchestration/jobs/conversion.py`, use
   `get_checkpoint_output_from_training_job` together with `MLClient` to resolve the
   checkpoint input to a proper asset reference like:

   ```python
   checkpoint_input = Input(
       type="uri_folder",
       path=f"azureml:{data_asset.name}:{data_asset.version}",
       mode="mount",
   )
   ```

   This avoids manually constructing datastore URIs and prevents
   `ScriptExecution.StreamAccess.NotFound` on empty or incorrect paths.

3. **Make the conversion script search nested `checkpoint/` directories**

   In `src/model_conversion/convert_to_onnx.py`, search both the root of the mounted checkpoint folder and
   any nested `checkpoint/` subdirectory for standard Hugging Face weight filenames (for
   example `pytorch_model.bin` or `model.safetensors`) before raising `FileNotFoundError`.

4. **Install required ONNX dependencies**

   Add compatible `onnx` and `onnxscript` versions to `config/environment/conda.yaml`:

   ```yaml
   dependencies:
     - numpy>=1.24.0,<2.0
     - onnx>=1.16.0
     - pip:
         - onnxscript>=0.1.0
   ```

5. **Improve ONNX export robustness and logging**

   In `convert_to_onnx`, call `torch.onnx.export` with explicit settings and
   detailed logging, for example:

   ```python
   torch.onnx.export(
       model,
       example_inputs,
       str(onnx_path),
       input_names=["input_ids", "attention_mask"],
       output_names=["logits"],
       dynamic_axes={
           "input_ids": {0: "batch", 1: "seq"},
           "attention_mask": {0: "batch", 1: "seq"},
           "logits": {0: "batch", 1: "seq"},
       },
       opset_version=18,
       do_constant_folding=True,
       dynamo=False,
   )
   ```

   Wrap export, quantization, and smoke tests in `try/except` blocks and log each
   phase with `print(..., flush=True)` or a helper like `_log("Starting FP32 ONNX export…")`
   so you can see long-running steps in `user_logs/std_log.txt`.

#### Prevention

- Keep the training output layout (`<output_dir>/checkpoint/`) and the conversion
  script’s search logic in sync whenever you change where checkpoints are saved.
- Always materialize the `checkpoint` output (even in smoke tests or dry runs) so the
  conversion job’s input data asset is not empty.
- Use asset references (`azureml:name:version`) for checkpoint inputs instead of
  hand-built datastore URIs.
- Pin compatible versions of `numpy`, `onnx`, and `onnxscript` in the environment.
- Use detailed, flushed logging in `convert_to_onnx.py` so you can quickly see
  where a conversion job is stuck.

## Prevention Checklist

Before submitting jobs, verify:

- [ ] Data assets use `azureml:name:version` references

- [ ] Batch sizes are appropriate for model size and VM memory

- [ ] All dependencies are pinned in `conda.yaml`

- [ ] AML placeholders use double braces `${{{{...}}}}`

- [ ] `SweepJob` uses correct parameter names

- [ ] Code snapshot includes all necessary directories (`code=".."`)

- [ ] Training script calls `mlflow.start_run()` for metric logging

## Related Documentation

- [Clean Code Principles](../docs/rules/CLEAN_CODE.md)

- [Documentation Guidelines](../docs/rules/CLEAN_DOC.md)

- [Training Script](../src/training/train.py)

- [Orchestration Notebook](../notebooks/01_orchestrate_training.ipynb)
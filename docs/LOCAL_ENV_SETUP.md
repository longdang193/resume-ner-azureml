# Local Conda Environment Setup

## Overview

**Purpose**: Create and activate a local Conda environment for the `resume-ner-azureml` project to run training scripts and notebooks locally before deploying to Google Colab or Azure ML.

**Audience**: Developers setting up the project for local development and testing.

**Platform**: Windows (macOS/Linux users should adjust paths accordingly).

---

## Prerequisites

- **Miniconda** or **Anaconda** installed
  - Default Windows path: `C:\Users\<YOUR_USER>\Miniconda3`
  - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not installed

---

## Quick Start

1. **Open a Conda-aware terminal** (see [Terminal Setup](#terminal-setup))
2. **Navigate to project root**:

   ```powershell
   cd path\to\resume-ner-azureml
   ```

3. **Create environment**:

   ```bash
   conda env create -f config/environment/conda.yaml
   ```

4. **Activate environment**:

   ```bash
   conda activate resume-ner-training
   ```

5. **Verify installation**:

   ```bash
   python --version
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

---

## Terminal Setup

### Option A: Miniconda/Anaconda Prompt (Recommended)

1. Open **"Miniconda Prompt"** or **"Anaconda Prompt"** from Windows Start menu
2. Navigate to project root:

   ```cmd
   cd path\to\resume-ner-azureml
   ```

3. `conda` commands work without additional setup

### Option B: PowerShell with Direct Path

If using PowerShell, call `conda.exe` directly:

```powershell
& "$env:USERPROFILE\Miniconda3\Scripts\conda.exe" env create -f config\environment\conda.yaml
```

> **Note**: Replace `Miniconda3` with your actual installation folder if different.

---

## Environment Creation

Create the environment from the project's Conda definition:

```bash
conda env create -f config/environment/conda.yaml
```

**What gets installed**:

- Python 3.10
- PyTorch (with CUDA support if available)
- Transformers, MLflow, Azure ML SDK, ONNX, and other dependencies

**Update existing environment**:

```bash
conda env update -f config/environment/conda.yaml
```

---

## Activation

### Miniconda/Anaconda Prompt

```bash
conda activate resume-ner-training
```

The prompt should show `(resume-ner-training)`.

### PowerShell

**Option 1 - Initialize Conda** (one-time, permanent):

```powershell
& "$env:USERPROFILE\Miniconda3\Scripts\conda.exe" init powershell
```

Close and reopen PowerShell, then:

```powershell
conda activate resume-ner-training
```

**Option 2 - Direct activation** (workaround):

```powershell
& "$env:USERPROFILE\Miniconda3\Scripts\activate.bat" resume-ner-training
```

---

## Verification

After activation, verify the environment:

```bash
python --version
# Expected: Python 3.10.x

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
# Expected: PyTorch version and CUDA availability status
```

---

## Local Test Run

Run a quick smoke test to verify the setup:

```bash
python src/training/train.py \
  --data-asset dataset_tiny \
  --config-dir config \
  --backbone distilbert \
  --epochs 1 \
  --batch-size 4
```

**Expected behavior**:

- Loads data from `dataset_tiny/`
- Trains for 1 epoch
- Saves outputs to `./outputs/`

**Success criteria**: Command completes without errors.

---

## Troubleshooting

### Error: `conda: The term 'conda' is not recognized`

**Cause**: Conda not on PATH.

**Solution**: Use [Miniconda/Anaconda Prompt](#option-a-minicondaanaconda-prompt-recommended) or call `conda.exe` directly (see [Terminal Setup](#option-b-powershell-with-direct-path)).

### Error: `conda-script.py: error: argument COMMAND: invalid choice: ''`

**Cause**: Wrong `conda` shim from `pip` is being used instead of real Conda.

**Solution**:

1. Use [Miniconda/Anaconda Prompt](#option-a-minicondaanaconda-prompt-recommended)
2. Or remove the pip shim: `pip uninstall conda`
3. Retry with the correct Conda executable

### Error: `CondaError: Run 'conda init' before 'conda activate'`

**Cause**: PowerShell not initialized for Conda.

**Solutions** (choose one):

1. **Initialize Conda** (see [PowerShell Option 1](#option-1---initialize-conda-one-time-permanent))
2. **Use direct activation** (see [PowerShell Option 2](#option-2---direct-activation-workaround))
3. **Use Miniconda Prompt** (see [Terminal Setup](#option-a-minicondaanaconda-prompt-recommended))

### Error: `Running cells with 'resume-ner-training (Python 3.10.19)' requires the ipykernel package`

**Cause**: `ipykernel` not installed in the Conda environment, preventing Jupyter notebooks from using it.

**Solution**: Install `ipykernel` in the environment:

**Using Miniconda/Anaconda Prompt**:
```bash
conda activate resume-ner-training
conda install ipykernel -y
```

**Using PowerShell with direct path**:
```powershell
& "$env:USERPROFILE\Miniconda3\Scripts\conda.exe" install -p "$env:USERPROFILE\Miniconda3\envs\resume-ner-training" ipykernel -y
```

**After installation**:
1. Reload VS Code window (`Ctrl+Shift+P` → "Developer: Reload Window")
2. Select the kernel: Click kernel selector in notebook → Choose "resume-ner-training (Python 3.10.19)"

---

## Next Steps

Once the environment is working:

- Run `notebooks/01_orchestrate_training_local.ipynb` via Jupyter or VS Code
- Validate training logic locally before deploying to Google Colab or Azure ML
- See [LOCAL_TRAINING.md](LOCAL_TRAINING.md) for local training workflow details

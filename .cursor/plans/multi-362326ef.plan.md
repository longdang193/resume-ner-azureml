<!-- 362326ef-2997-46e5-b774-c64f279a0d63 c4e80223-9f67-42a5-aa65-3a4853b1366a -->
# Multi-GPU Training & HPO Plan

### Goals

- Add **optional multi-GPU (DDP) support** for both:
- Final training step (single best config)
- HPO sweeps (local Optuna, including k-fold CV)
- Keep **current single-GPU flow as the default**; auto-detect and enable DDP only when safe (e.g., `cuda` and `device_count > 1`).
- Support **Kaggle 2×T4** and general multi-GPU machines without breaking Colab/single-GPU.
- Enforce **SRP and Clean Code** so each module has a focused responsibility, and
all tunable behavior is centralized in YAML configs.

### High-Level Design (SRP & Centralized Config)

- **Centralize all knobs in YAML** only:
- Extend existing YAMLs (`config/train.yaml`, `config/env/*.yaml`, `config/hpo/*.yaml`) with a `distributed` section (e.g. `enabled`, `backend`, `world_size`, `grad_accumulation`, `sync_batchnorm`).
- Avoid hard-coding DDP/device choices in notebooks or scripts; they should only read resolved config objects from `orchestration.config_loader`.
- **Clear separation of responsibilities**:
- `training/config.py` (and config loader): parse and validate configs, including `distributed` settings → return plain Python structures.
- `training/distributed.py` (new): encapsulate DDP-specific concerns:
- Env detection (`cuda`, `device_count`).
- Construction of a `RunContext` (single-GPU vs DDP) with `world_size`, `rank`, `device`, and init/teardown of `torch.distributed`.
- Creation of `DistributedSampler` where needed.
- `trainer.py`: own the **training loop only**:
- Given a `RunContext`, model, data loaders, and hyperparameters, run epochs/steps and report metrics.
- No direct CLI parsing, no device counting, no DDP init.
- CLI / orchestration entrypoints (`train.py`, local sweeps, notebooks):
- Gather configs via the centralized loader.
- Choose which `RunContext` to use via `training.distributed`.
- Call trainer with the prepared context.
- **Orchestration for HPO + final training**:
- For both HPO and final training, the orchestration layer:
- Resolves configs (including `distributed`) via YAML loader.
- Calls a single "train_once(config, run_context)" style function; the only difference is the context (single-GPU vs DDP) and the hyperparameters passed.
- Optuna objectives become thin: sample params → merge into config → call trainer with selected `RunContext`.
- **Graceful fallback**:
- If `distributed.enabled: true` but the environment only exposes 1 GPU or DDP init fails, `training.distributed` transparently returns a single-GPU `RunContext`.
- All DDP-specific imports and error handling stay inside `training.distributed` and do not leak into business logic or notebooks.

### Steps

1. **Config & Environment Detection (Centralized YAML)**

- Extend `config/train.yaml` and/or `config/env/*.yaml` with a `distributed` section, e.g.:
- `distributed.enabled: false`
- `distributed.backend: nccl`
- `distributed.world_size: auto`
- Optional: `distributed.sync_batchnorm`, `distributed.fp16`.
- Update `orchestration.config_loader` / `training/config.py` to:
- Validate `distributed` options (types, allowed values).
- Provide a small `ResolvedDistributedConfig` object used by `training.distributed`.

2. **Implement Core DDP & Context Abstractions (SRP in training)**

- Create `training/distributed.py` with:
- `detect_hardware()` to return `cuda_available`, `device_count`, etc.
- `create_run_context(distributed_cfg, hardware)` which returns either:
- `SingleProcessContext(device="cuda" or "cpu")`, or
- `DDPContext(world_size, rank, local_rank, backend, device, init/cleanup hooks)`.
- Helpers to build `DataLoader` samplers (standard or `DistributedSampler`).
- In `trainer.py`:
- Refactor to a function like `run_training(model, dataloaders, cfg, context)` which:
- Uses `context.device` to move tensors/models.
- Uses samplers provided/selected outside.
- Only rank 0 handles metrics/logging when `context` is DDP.
- If a CLI entrypoint is used, keep it thin: parse args → load YAMLs → build configs → create context → call `run_training`.

3. **Wire DDP into Final Training Orchestration (Notebooks & train.py)**

- In `01_orchestrate_training_local.ipynb` and `01_orchestrate_training_colab.ipynb`:
- Keep responsibility to: load configs, build experiment metadata, invoke `train.py` with minimal flags.
- Add a small environment cell that prints `cuda` status and `device_count` (for visibility) but does not decide training logic.
- When launching final training via subprocess:
- Pass a `--distributed-config` or rely on the centralized YAML only (preferred) so `train.py` does not receive extra device flags beyond what configs already specify.
- In `train.py`:
- Load configs via `orchestration.config_loader`.
- Ask `training.distributed.create_run_context` for a context.
- Call `trainer.run_training` with that context.

4. **Wire DDP into Local HPO Sweeps (SRP for objectives)**

- In `src/orchestration/jobs/local_sweeps.py`:
- Ensure the Optuna objective functions:
- Only sample hyperparameters and merge them into a base config from YAML.
- Call a shared helper like `run_single_trial(config)` which:
- Uses `training.distributed.create_run_context` to decide DDP vs single-GPU.
- Calls `trainer.run_training` once.
- For k-fold CV: the CV logic remains in `cv_utils` and calls the same `run_single_trial` per fold; the context decides whether each fold run is DDP or single-GPU.

5. **Kaggle/Colab Environment Guardrails (Clean orchestration)**

- Add a small utility (could live in `shared/env_utils.py`) to:
- Detect platform (`local`, `colab`, `kaggle`).
- Optionally override certain `distributed` settings (e.g., `backend` on Kaggle).
- In notebooks, use this only to print status and maybe tweak config before passing to the trainer, but do not embed training logic.

6. **Testing & Validation**

- **Single-GPU regression check**:
- Run smoke HPO and final training on a single GPU (local or Colab) and confirm metrics and behavior match pre-DDP implementation.
- **Multi-GPU functional tests**:
- On a multi-GPU box or Kaggle 2×T4 where both GPUs are visible:
- Run a short final training and confirm both GPUs are used (via logs / `nvidia-smi`).
- Run a short HPO sweep with DDP enabled and confirm each trial completes correctly.
- Ensure:
- MLflow logging works with `rank 0` only.
- Checkpoints aren’t duplicated by multiple ranks.
- Fallback to single-GPU works if DDP init fails.

7. **Documentation & Config Examples**

- Update `docs/LOCAL_TRAINING.md` / `docs/Centralized_Configuration.md` / new `docs/MULTI_GPU_TRAINING.md` to:
- Document the `distributed` YAML block and its options.
- Show examples for:
- Single-GPU (default).
- Multi-GPU on Kaggle/VM (e.g., `distributed.enabled: true`).
- Explain SRP-style architecture: where configs live, what `trainer` vs `distributed` vs notebooks do.

### Implementation Notes

- Start with **final training** (single config) using the new `RunContext` abstraction.
- Once stable, route HPO and k-fold CV through the same `run_single_trial` helper for consistency and reduced duplication.
- Keep multi-GPU behavior completely driven by YAML + environment detection to avoid scattered flags and improve maintainability.

### To-dos

- [ ] Extend train/env configs with a distributed section and implement a helper to detect multi-GPU and decide use_ddp/world_size.
- [ ] Implement DDP worker and launcher functions in trainer.py with DistributedDataParallel and DistributedSampler.
- [ ] Update final training orchestration (train.py and notebooks) to optionally call the DDP launcher based on config/env.
- [ ] Update local HPO sweeps to optionally run trials via DDP-enabled training, ensuring one DDP job per trial.
- [ ] Add environment detection (Kaggle vs Colab vs local), choose backends, and validate both single- and multi-GPU flows.
- [ ] Document multi-GPU usage, config knobs, and environment-specific guidance in project docs.
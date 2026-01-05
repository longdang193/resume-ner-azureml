"""Distributed training helpers and run context abstractions.

This module is intentionally focused on environment and context management
for training (SRP). It does *not* implement the training loop itself; that
responsibility lives in `trainer.py`.

The initial version provides a SingleProcessContext abstraction and a
factory to resolve a context from YAML-driven distributed config plus
hardware detection. DDP-specific wiring can be added here later without
touching business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from .config import ResolvedDistributedConfig


@dataclass
class RunContext:
    """Base run context for training.

    Attributes:
        device: torch.device used for this process.
        world_size: Total number of processes participating in training.
        rank: Global rank of this process.
        local_rank: Local rank on the current node.
        distributed: Whether this context represents a distributed run.
    """

    device: torch.device
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    distributed: bool = False

    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        """Synchronize processes if distributed; no-op for single process."""
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()


class SingleProcessContext(RunContext):
    """Context for standard single-process (single-GPU or CPU) training."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(
            device=device,
            world_size=1,
            rank=0,
            local_rank=0,
            distributed=False,
        )


@dataclass
class DDPContext(RunContext):
    """Context for multi-process Distributed Data Parallel training.

    This does not itself initialize the process group; that is handled by
    `init_process_group_if_needed`, allowing cleaner separation of concerns.
    """

    backend: str = "nccl"
    init_method: str = "env://"
    timeout_seconds: int = 1800


def detect_hardware() -> Tuple[bool, int]:
    """Detect CUDA availability and number of visible GPUs."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    return cuda_available, device_count


def is_multi_gpu_available() -> bool:
    """Return True if at least two CUDA devices are visible."""
    cuda_available, device_count = detect_hardware()
    return cuda_available and device_count > 1


def _read_rank_env() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Read rank-related environment variables set by torchrun/launchers.

    Returns (world_size, rank, local_rank) where any missing value is None.
    """
    def _get_int(name: str) -> Optional[int]:
        value = os.getenv(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    world_size = _get_int("WORLD_SIZE")
    rank = _get_int("RANK")
    local_rank = _get_int("LOCAL_RANK")
    return world_size, rank, local_rank


def should_use_ddp(dist_cfg: ResolvedDistributedConfig, device_count: int) -> bool:
    """Decide whether DDP *should* be used based on config + hardware.

    This does not check environment variables; it simply encodes the policy:
    - distributed.enabled must be true
    - at least 2 GPUs must be visible
    """
    return bool(dist_cfg.enabled and device_count > 1)


def init_process_group_if_needed(context: RunContext) -> None:
    """Initialize torch.distributed process group for a DDPContext.

    Safe to call multiple times; if the process group is already initialized
    or the context is not distributed, this is a no-op.
    """
    if not isinstance(context, DDPContext):
        return

    if not dist.is_available() or dist.is_initialized():
        return

    timeout = timedelta(seconds=context.timeout_seconds)
    dist.init_process_group(
        backend=context.backend,
        init_method=context.init_method,
        world_size=context.world_size,
        rank=context.rank,
        timeout=timeout,
    )


def create_run_context(dist_cfg: ResolvedDistributedConfig) -> RunContext:
    """Create a RunContext from resolved distributed config and hardware.

    Behavior:
    - If CUDA is unavailable → CPU SingleProcessContext.
    - If CUDA is available but dist_cfg.disabled or only 1 GPU → cuda:0 SingleProcessContext.
    - If CUDA is available, dist_cfg.enabled and >=2 GPUs *and* rank env vars present → DDPContext.
    - If rank env vars are missing but dist_cfg.enabled → fall back to SingleProcessContext.
    """
    cuda_available, device_count = detect_hardware()

    if not cuda_available or device_count == 0:
        # CPU-only fallback
        device = torch.device("cpu")
        return SingleProcessContext(device=device)

    # Decide if DDP is even desired.
    if not should_use_ddp(dist_cfg, device_count):
        # Single-GPU CUDA context on device 0.
        device = torch.device("cuda:0")
        return SingleProcessContext(device=device)

    # At this point, config + hardware say \"yes\" to DDP, but we still require
    # rank information (typically provided by torchrun or a launcher).
    world_size_env, rank_env, local_rank_env = _read_rank_env()

    if world_size_env is None or rank_env is None or local_rank_env is None:
        # Misconfigured DDP: log-friendly fallback to single process on cuda:0.
        # The launcher is responsible for surfacing a clear message if needed.
        device = torch.device("cuda:0")
        return SingleProcessContext(device=device)

    device = torch.device(f"cuda:{local_rank_env}")
    return DDPContext(
        device=device,
        world_size=world_size_env,
        rank=rank_env,
        local_rank=local_rank_env,
        distributed=True,
        backend=dist_cfg.backend,
        init_method=dist_cfg.init_method,
        timeout_seconds=dist_cfg.timeout_seconds,
    )

"""Distributed training launcher for DDP setup."""

import os
import argparse
import torch.multiprocessing as mp
from pathlib import Path

from training.config import build_training_config, resolve_distributed_config
from training.distributed import detect_hardware, should_use_ddp
from training.orchestrator import run_training
from common.shared.argument_parsing import validate_config_dir


def _ddp_worker(local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    """
    DDP worker entrypoint used with torch.multiprocessing.spawn.
    
    This sets rank-related environment variables and then delegates to
    `run_training`, which performs DDP initialization and training.
    
    Args:
        local_rank: Local rank of this process.
        world_size: Total number of processes.
        args: Parsed command-line arguments.
    """
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(local_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    # Ensure env:// rendezvous has required address/port when using spawn.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    run_training(args)


def launch_training(args: argparse.Namespace) -> None:
    """
    Launch training in single-process or DDP mode.
    
    Args:
        args: Parsed command-line arguments.
    """
    # Build config once here to decide between single-process and DDP modes.
    config_dir = validate_config_dir(args.config_dir)
    
    config = build_training_config(args, config_dir)
    dist_cfg = resolve_distributed_config(config)
    _, device_count = detect_hardware()
    
    # If already running under torchrun (WORLD_SIZE set), do not spawn again.
    world_size_env = os.getenv("WORLD_SIZE")
    
    if world_size_env is None and should_use_ddp(dist_cfg, device_count):
        # Decide world_size: explicit integer from config or all visible GPUs.
        world_size = dist_cfg.world_size or device_count
        if world_size < 2:
            # Fallback to single-process if config/hardware are inconsistent.
            run_training(args, prebuilt_config=config)
            return
        
        mp.spawn(
            _ddp_worker,
            nprocs=world_size,
            args=(world_size, args),
        )
    else:
        # Single-process path (CPU, single GPU, or torchrun-managed ranks).
        run_training(args, prebuilt_config=config)


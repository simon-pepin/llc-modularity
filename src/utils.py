"""Utilities for logging, seeding, device management, and config handling."""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """Save config to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist, return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

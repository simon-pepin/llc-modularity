"""Utilities for logging, seeding, device management, config handling, and plotting."""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
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


def plot_training_curves(
    results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot per-subtask train and test loss curves from training results.

    Args:
        results: Dict returned by train_cmsp, containing eval_steps,
            subtask_losses, and test_subtask_losses.
        save_path: If provided, save the figure to this path.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure.
    """
    eval_steps = results["eval_steps"]
    train_losses = results["subtask_losses"]
    test_losses = results.get("test_subtask_losses", {})

    subtask_names = list(train_losses.keys())
    has_test = bool(test_losses)

    ncols = 2 if has_test else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)

    # Train loss panel
    ax_train = axes[0, 0]
    for name in subtask_names:
        losses = train_losses[name]
        ax_train.plot(eval_steps[: len(losses)], losses, label=name)
    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Loss")
    ax_train.set_title("Train Loss (fresh samples)")
    ax_train.set_xscale("log")
    ax_train.set_yscale("log")
    ax_train.legend(fontsize=8)

    # Test loss panel
    if has_test:
        ax_test = axes[0, 1]
        for name in subtask_names:
            losses = test_losses[name]
            ax_test.plot(eval_steps[: len(losses)], losses, label=name)
        ax_test.set_xlabel("Step")
        ax_test.set_ylabel("Loss")
        ax_test.set_title("Test Loss (fixed dataset)")
        ax_test.set_xscale("log")
        ax_test.set_yscale("log")
        ax_test.legend(fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig

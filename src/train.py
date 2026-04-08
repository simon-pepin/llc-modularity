"""Training loop for CMSP models with per-subtask loss tracking and checkpointing.

Follows the protocol from ejmichaud/narrow: joint training on atomic + composite
subtasks, Adam optimizer, per-subtask loss monitoring.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import code_to_str, generate_cmsp_batch, make_subtask_indices
from .model import count_parameters, make_mlp
from .utils import ensure_dir, save_config, set_seed


def compute_per_subtask_loss(
    model: nn.Module,
    n: int,
    m: int,
    subtask_indices: List[List[int]],
    task_codes: List[List[int]],
    eval_samples: int,
    loss_fn: nn.Module,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, float]:
    """Evaluate the model's loss on each subtask separately."""
    model.eval()
    losses = {}
    with torch.no_grad():
        for code in task_codes:
            x, y = generate_cmsp_batch(
                n=n, m=m, subtask_indices=subtask_indices,
                task_codes=[code], batch_sizes=[eval_samples],
                device=device, dtype=dtype,
            )
            logits = model(x)
            loss = loss_fn(logits, y).item()
            losses[code_to_str(code)] = loss
    model.train()
    return losses


def train_cmsp(
    config: Dict[str, Any],
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train an MLP on the CMSP task.

    Args:
        config: Training configuration dict with keys:
            - n: task bits (default: inferred from m*k)
            - m: number of atomic subtasks (default: 4)
            - k: bits per subtask (default: 4)
            - width: hidden layer width (default: 128)
            - depth: network depth (default: 3)
            - activation: activation function (default: 'relu')
            - use_layernorm: whether to use LayerNorm (default: False)
            - task_codes: list of task codes (default: atomics + one composite)
            - samples_per_task: samples per code per step (default: 2000)
            - steps: total training steps (default: 200000)
            - lr: learning rate (default: 1e-3)
            - seed: random seed (default: 0)
            - dtype: 'float32' or 'float64' (default: 'float32')
            - device: torch device (default: 'cuda' if available)
            - eval_every: evaluate per-subtask loss every N steps (default: 100)
            - checkpoint_every: save checkpoint every N steps (default: 0, disabled)
            - checkpoint_steps: specific steps to save checkpoints (default: [])
        save_dir: Directory to save results and checkpoints. None = don't save.
        verbose: Whether to show progress bar.

    Returns:
        Dict with keys: model, config, steps, losses, subtask_losses,
        subtask_indices, n_parameters.
    """
    # Parse config with defaults
    m = config.get("m", 4)
    k = config.get("k", 4)
    n = config.get("n", m * k)
    width = config.get("width", 128)
    depth = config.get("depth", 3)
    activation = config.get("activation", "relu")
    use_layernorm = config.get("use_layernorm", False)
    samples_per_task = config.get("samples_per_task", 2000)
    steps = config.get("steps", 200_000)
    lr = config.get("lr", 1e-3)
    seed = config.get("seed", 0)
    dtype_str = config.get("dtype", "float32")
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    eval_every = config.get("eval_every", 100)
    checkpoint_every = config.get("checkpoint_every", 0)
    checkpoint_steps = set(config.get("checkpoint_steps", []))

    dtype = torch.float32 if dtype_str == "float32" else torch.float64

    # Default task codes: all atomics + one composite of all
    task_codes = config.get("task_codes", None)
    if task_codes is None:
        task_codes = [[i] for i in range(m)] + [list(range(m))]

    set_seed(seed)

    # Setup
    subtask_indices = config.get("subtask_indices", None)
    if subtask_indices is None:
        subtask_indices = make_subtask_indices(m, k)

    model = make_mlp(n=n, m=m, width=width, depth=depth,
                     activation=activation, use_layernorm=use_layernorm)
    model = model.to(dtype).to(device)
    n_params = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Tracking
    step_list = []
    loss_list = []
    subtask_losses = {code_to_str(c): [] for c in task_codes}
    eval_steps = []

    if save_dir is not None:
        save_dir = ensure_dir(save_dir)
        save_config(config, save_dir / "config.yaml")

    batch_sizes = [samples_per_task] * len(task_codes)
    iterator = tqdm(range(steps), desc="Training", disable=not verbose)

    for step in iterator:
        # Evaluation
        if step % eval_every == 0:
            per_task = compute_per_subtask_loss(
                model, n, m, subtask_indices, task_codes,
                eval_samples=samples_per_task, loss_fn=loss_fn,
                device=device, dtype=dtype,
            )
            for key, val in per_task.items():
                subtask_losses[key].append(val)
            eval_steps.append(step)

            if verbose:
                task_str = " | ".join(f"{k}:{v:.4f}" for k, v in per_task.items())
                iterator.set_postfix_str(task_str[:80])

        # Training step
        model.train()
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=subtask_indices,
            task_codes=task_codes, batch_sizes=batch_sizes,
            device=device, dtype=dtype,
        )
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_list.append(step)
        loss_list.append(loss.item())

        # Checkpointing
        should_checkpoint = (
            (checkpoint_every > 0 and step > 0 and step % checkpoint_every == 0)
            or step in checkpoint_steps
        )
        if should_checkpoint and save_dir is not None:
            ckpt_dir = ensure_dir(save_dir / "checkpoints" / f"step_{step}")
            torch.save(model.state_dict(), ckpt_dir / "model.pt")

    # Final evaluation
    per_task = compute_per_subtask_loss(
        model, n, m, subtask_indices, task_codes,
        eval_samples=samples_per_task, loss_fn=loss_fn,
        device=device, dtype=dtype,
    )
    for key, val in per_task.items():
        subtask_losses[key].append(val)
    eval_steps.append(steps)

    # Save final model and results
    results = {
        "config": config,
        "steps": step_list,
        "eval_steps": eval_steps,
        "losses": loss_list,
        "subtask_losses": subtask_losses,
        "subtask_indices": subtask_indices,
        "task_codes": task_codes,
        "n_parameters": n_params,
        "final_subtask_losses": per_task,
    }

    if save_dir is not None:
        torch.save(model.state_dict(), save_dir / "model.pt")
        with open(save_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)

    results["model"] = model
    return results

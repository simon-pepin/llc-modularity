"""Data-restricted LLC estimation for additivity experiments.

The core idea: estimate the LLC using SGLD sampling with loss functions computed
over different subsets of the data (corresponding to different subtasks). This
gives us lambda_hat(T_i) for each subtask T_i, all starting from the same
jointly-trained weights.
"""

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from devinterp.optim import SGLD
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import estimate_learning_coeff_with_summary


def cross_entropy_evaluate(model: nn.Module, batch) -> torch.Tensor:
    """Default evaluation function: cross-entropy loss on (x, y) batch."""
    x, y = batch
    logits = model(x)
    return F.cross_entropy(logits, y)


def compute_init_loss(
    model: nn.Module,
    dataloader: DataLoader,
    evaluate: Callable = cross_entropy_evaluate,
    device: str = "cpu",
    n_batches: int = 10,
) -> float:
    """Compute the average loss at current weights over n_batches.

    This is needed as init_loss for LLCEstimator. Must be computed on the
    same restricted dataset used for SGLD.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            x, y = batch
            x, y = x.to(device), y.to(device)
            loss = evaluate(model, (x, y))
            total_loss += loss.item()
            count += 1
    model.train()
    if count == 0:
        raise ValueError("Dataloader yielded no batches")
    return total_loss / count


def compute_nbeta(dataloader: DataLoader) -> float:
    """Compute nbeta = effective_n / log(effective_n).

    effective_n is the batch size (what SGLD sees per step).
    This matches devinterp's default_nbeta logic.
    """
    batch_size = dataloader.batch_size
    if batch_size is None or batch_size <= 1:
        return 1.0
    return batch_size / math.log(batch_size)


def estimate_llc(
    model: nn.Module,
    dataloader: DataLoader,
    num_chains: int = 10,
    num_draws: int = 500,
    num_burnin_steps: int = 100,
    num_steps_bw_draws: int = 1,
    learning_rate: float = 1e-4,
    localization: float = 0.0,
    device: str = "cuda",
    nbeta: Optional[float] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Estimate LLC at the current model weights using the given dataloader.

    The dataloader determines WHICH loss function is used for SGLD sampling.
    Passing a subtask-restricted dataloader gives the data-restricted LLC.

    Args:
        model: Trained model (weights will not be modified).
        dataloader: Data for computing the loss during SGLD.
        num_chains: Number of independent SGLD chains.
        num_draws: Number of samples per chain (after burn-in).
        num_burnin_steps: Number of initial steps to discard.
        num_steps_bw_draws: Steps between consecutive draws.
        learning_rate: SGLD learning rate.
        localization: Strength of pull back to initialization.
        device: Torch device.
        nbeta: Inverse temperature. If None, computed from batch size.
        seed: Random seed.
        verbose: Show progress.

    Returns:
        Dict with keys: llc_mean, llc_std, llc_per_chain, loss_trace, init_loss, nbeta.
    """
    # Work on a copy so we don't modify the original
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()

    if nbeta is None:
        nbeta = compute_nbeta(dataloader)

    init_loss = compute_init_loss(
        model_copy, dataloader, evaluate=cross_entropy_evaluate,
        device=device, n_batches=max(num_chains, 10),
    )

    # Use devinterp's high-level API
    results = estimate_learning_coeff_with_summary(
        model=model_copy,
        loader=dataloader,
        evaluate=cross_entropy_evaluate,
        sampling_method=SGLD,
        sampling_method_kwargs=dict(
            lr=learning_rate,
            nbeta=nbeta,
            localization=localization,
        ),
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        init_loss=init_loss,
        device=device,
        seed=seed,
        verbose=verbose,
        online=False,
    )

    # Extract per-chain values
    llc_per_chain = []
    for i in range(num_chains):
        key = f"llc-chain/{i}"
        if key in results:
            llc_per_chain.append(results[key])

    return {
        "llc_mean": results["llc/mean"],
        "llc_std": results["llc/std"],
        "llc_per_chain": llc_per_chain,
        "loss_trace": results.get("loss/trace", None),
        "init_loss": init_loss,
        "nbeta": nbeta,
    }


def estimate_subtask_llcs(
    model: nn.Module,
    subtask_dataloaders: Dict[str, DataLoader],
    device: str = "cuda",
    verbose: bool = True,
    **llc_kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Estimate LLC separately for each subtask's data.

    For each key in subtask_dataloaders, runs SGLD using only that subtask's
    loss function, starting from the SAME jointly-trained weights.

    Args:
        model: Jointly-trained model.
        subtask_dataloaders: Dict mapping subtask name (e.g. '{0}', '{0,1}')
            to a DataLoader containing only that subtask's data.
        device: Torch device.
        verbose: Show progress.
        **llc_kwargs: Additional kwargs passed to estimate_llc.

    Returns:
        Dict mapping subtask name to LLC estimation results.
    """
    results = {}
    for name, loader in subtask_dataloaders.items():
        if verbose:
            print(f"\n--- Estimating LLC for subtask {name} ---")
        results[name] = estimate_llc(
            model=model,
            dataloader=loader,
            device=device,
            verbose=verbose,
            **llc_kwargs,
        )
        if verbose:
            r = results[name]
            print(f"    LLC = {r['llc_mean']:.4f} ± {r['llc_std']:.4f}")
    return results

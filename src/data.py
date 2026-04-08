"""CMSP (Compositional Multitask Sparse Parity) data generation.

Adapted from ejmichaud/narrow. Generates multitask sparse parity data where:
- There are m atomic subtasks, each defined by a disjoint index set of k bits.
- Control bits select which subtask(s) are active.
- Labels are parity of task bits at the union of active subtask index sets.
"""

import itertools
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset


def generate_cmsp_batch(
    n: int,
    m: int,
    subtask_indices: List[List[int]],
    task_codes: List[List[int]],
    batch_sizes: List[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Generate a batch of CMSP samples.

    Args:
        n: Number of task bits.
        m: Number of atomic subtasks (= number of control bits).
        subtask_indices: List of m index lists, each specifying which task bits
            belong to atomic subtask i. Must be disjoint.
        task_codes: List of codes, where each code is a list of active control
            bit indices. E.g. [[0], [1], [0,1,2,3]].
        batch_sizes: Number of samples to generate per code.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        x: Input tensor of shape (total_batch, m + n). First m dims are control
           bits, remaining n dims are task bits.
        y: Label tensor of shape (total_batch,), dtype int64. Binary parity labels.
    """
    assert len(subtask_indices) == m
    assert len(task_codes) == len(batch_sizes)

    total = sum(batch_sizes)
    x = torch.zeros((total, m + n), dtype=dtype, device=device)
    bits = torch.randint(0, 2, (total, n), dtype=dtype, device=device)
    x[:, m:] = bits

    y = torch.empty((total,), dtype=torch.int64, device=device)

    idx = 0
    for code, size in zip(task_codes, batch_sizes):
        if size <= 0:
            continue
        # Union of bit positions for active subtasks
        S = sorted(set(itertools.chain.from_iterable(subtask_indices[c] for c in code)))
        # Set control bits
        for c in code:
            x[idx : idx + size, c] = 1
        # Compute parity label
        parity_bits = bits[idx : idx + size][:, S]
        y[idx : idx + size] = parity_bits.sum(dim=1).remainder(2).to(torch.int64)
        idx += size

    return x, y


def make_subtask_indices(m: int, k: int) -> List[List[int]]:
    """Create m disjoint subtask index sets, each of size k.

    Subtask i uses bits [i*k, ..., (i+1)*k - 1].
    Total number of task bits n = m * k.
    """
    return [list(range(i * k, (i + 1) * k)) for i in range(m)]


def code_to_str(code: List[int]) -> str:
    """Convert a task code like [0, 1, 2] to a string key '{0,1,2}'."""
    return "{" + ",".join(str(c) for c in sorted(code)) + "}"


def str_to_code(s: str) -> List[int]:
    """Convert a string key like '{0,1,2}' back to [0, 1, 2]."""
    inner = s.strip("{}")
    if not inner:
        return []
    return [int(x) for x in inner.split(",")]


class CMSPDataset(Dataset):
    """Fixed-size dataset of CMSP samples for a specific set of subtasks.

    Pre-generates all samples at construction time. This is needed for LLC
    estimation, where we want a fixed dataset (not freshly sampled each time).
    """

    def __init__(
        self,
        n: int,
        m: int,
        subtask_indices: List[List[int]],
        task_codes: List[List[int]],
        samples_per_code: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n: Number of task bits.
            m: Number of atomic subtasks.
            subtask_indices: Index sets for each atomic subtask.
            task_codes: Which subtask combinations to include.
            samples_per_code: Number of samples per code.
            device: Torch device.
            dtype: Torch dtype.
            seed: Random seed for reproducibility.
        """
        self.n = n
        self.m = m
        self.subtask_indices = subtask_indices
        self.task_codes = task_codes
        self.samples_per_code = samples_per_code

        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            # Generate on CPU then move, for reproducibility across devices
            old_state = torch.random.get_rng_state()
            torch.manual_seed(seed)

        batch_sizes = [samples_per_code] * len(task_codes)
        self.x, self.y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=subtask_indices,
            task_codes=task_codes, batch_sizes=batch_sizes,
            device=device, dtype=dtype,
        )

        # Track which code each sample belongs to
        self.code_labels = []
        for i, (code, size) in enumerate(zip(task_codes, batch_sizes)):
            self.code_labels.extend([i] * size)
        self.code_labels = torch.tensor(self.code_labels, device=device)

        if seed is not None:
            torch.random.set_rng_state(old_state)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_subtask_dataloaders(
    n: int,
    m: int,
    subtask_indices: List[List[int]],
    task_codes: List[List[int]],
    samples_per_code: int,
    batch_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Create a dataloader for each individual code, plus composite unions.

    For LLC estimation, we need dataloaders restricted to specific subtask data.
    This creates one dataloader per code, keyed by code string (e.g. '{0}', '{0,1}').

    Args:
        n, m, subtask_indices: CMSP task parameters.
        task_codes: All codes to generate data for.
        samples_per_code: Samples per code.
        batch_size: Batch size for dataloaders.
        device, dtype: Torch parameters.
        seed: Random seed.

    Returns:
        Dict mapping code string to DataLoader containing only that code's samples.
    """
    dataloaders = {}

    for i, code in enumerate(task_codes):
        key = code_to_str(code)
        ds = CMSPDataset(
            n=n, m=m, subtask_indices=subtask_indices,
            task_codes=[code], samples_per_code=samples_per_code,
            device=device, dtype=dtype,
            seed=(seed + i) if seed is not None else None,
        )
        dataloaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloaders


def make_joint_dataloader(
    n: int,
    m: int,
    subtask_indices: List[List[int]],
    task_codes: List[List[int]],
    samples_per_code: int,
    batch_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> DataLoader:
    """Create a single dataloader with samples from all codes mixed together.

    Used for training (joint loss over all subtasks).
    """
    ds = CMSPDataset(
        n=n, m=m, subtask_indices=subtask_indices,
        task_codes=task_codes, samples_per_code=samples_per_code,
        device=device, dtype=dtype, seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def make_union_dataloaders(
    n: int,
    m: int,
    subtask_indices: List[List[int]],
    code_groups: Dict[str, List[List[int]]],
    samples_per_code: int,
    batch_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Create dataloaders for arbitrary unions of codes.

    For additivity measurement, we need dataloaders for unions like
    {0} ∪ {1} (data from both atomic subtask 0 and 1 combined).

    Args:
        code_groups: Dict mapping a label (e.g. '{0}∪{1}') to a list of codes
            whose data should be unioned.
        Other args: same as make_subtask_dataloaders.

    Returns:
        Dict mapping group label to DataLoader.
    """
    dataloaders = {}
    for i, (label, codes) in enumerate(code_groups.items()):
        ds = CMSPDataset(
            n=n, m=m, subtask_indices=subtask_indices,
            task_codes=codes, samples_per_code=samples_per_code,
            device=device, dtype=dtype,
            seed=(seed + 1000 + i) if seed is not None else None,
        )
        dataloaders[label] = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dataloaders

"""Tests for CMSP data generation."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    CMSPDataset,
    code_to_str,
    generate_cmsp_batch,
    make_joint_dataloader,
    make_subtask_dataloaders,
    make_subtask_indices,
    make_union_dataloaders,
    str_to_code,
)


class TestSubtaskIndices:
    def test_basic(self):
        indices = make_subtask_indices(m=4, k=3)
        assert len(indices) == 4
        assert indices[0] == [0, 1, 2]
        assert indices[1] == [3, 4, 5]
        assert indices[2] == [6, 7, 8]
        assert indices[3] == [9, 10, 11]

    def test_disjoint(self):
        indices = make_subtask_indices(m=4, k=4)
        all_bits = set()
        for idx_set in indices:
            for bit in idx_set:
                assert bit not in all_bits, f"Bit {bit} appears in multiple subtasks"
                all_bits.add(bit)


class TestCodeConversion:
    def test_code_to_str(self):
        assert code_to_str([0]) == "{0}"
        assert code_to_str([0, 1, 2]) == "{0,1,2}"
        assert code_to_str([3, 1]) == "{1,3}"  # sorted

    def test_str_to_code(self):
        assert str_to_code("{0}") == [0]
        assert str_to_code("{0,1,2}") == [0, 1, 2]

    def test_roundtrip(self):
        code = [0, 2, 3]
        assert str_to_code(code_to_str(code)) == sorted(code)


class TestGenerateBatch:
    def test_shape(self):
        m, n = 4, 16
        indices = make_subtask_indices(m=4, k=4)
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0], [1]], batch_sizes=[100, 200],
        )
        assert x.shape == (300, m + n)
        assert y.shape == (300,)

    def test_control_bits(self):
        """Control bits should be set correctly for each subtask."""
        m, n = 4, 16
        indices = make_subtask_indices(m=4, k=4)
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0], [2]], batch_sizes=[50, 50],
        )
        # First 50 samples: control bit 0 = 1, others = 0
        assert (x[:50, 0] == 1).all()
        assert (x[:50, 1] == 0).all()
        assert (x[:50, 2] == 0).all()
        assert (x[:50, 3] == 0).all()
        # Next 50: control bit 2 = 1
        assert (x[50:, 2] == 1).all()
        assert (x[50:, 0] == 0).all()

    def test_composite_control_bits(self):
        """Composite task should have multiple control bits set."""
        m, n = 4, 16
        indices = make_subtask_indices(m=4, k=4)
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0, 1, 2, 3]], batch_sizes=[100],
        )
        assert (x[:, 0] == 1).all()
        assert (x[:, 1] == 1).all()
        assert (x[:, 2] == 1).all()
        assert (x[:, 3] == 1).all()

    def test_parity_correctness_single(self):
        """Verify parity labels for a single atomic task."""
        m, n, k = 2, 6, 3
        indices = [[0, 1, 2], [3, 4, 5]]
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0]], batch_sizes=[1000],
        )
        # Manually compute parity of bits 0, 1, 2
        task_bits = x[:, m:]  # shape (1000, 6)
        expected_parity = task_bits[:, 0:3].sum(dim=1).remainder(2).long()
        assert (y == expected_parity).all()

    def test_parity_correctness_composite(self):
        """Verify parity labels for a composite task (union of index sets)."""
        m, n, k = 2, 6, 3
        indices = [[0, 1, 2], [3, 4, 5]]
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0, 1]], batch_sizes=[1000],
        )
        # Composite of {0,1}: parity of ALL 6 bits
        task_bits = x[:, m:]
        expected_parity = task_bits.sum(dim=1).remainder(2).long()
        assert (y == expected_parity).all()

    def test_labels_are_binary(self):
        m, n = 4, 16
        indices = make_subtask_indices(m=4, k=4)
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0], [1], [0, 1, 2, 3]], batch_sizes=[500, 500, 500],
        )
        assert set(y.tolist()).issubset({0, 1})

    def test_task_bits_are_binary(self):
        m, n = 4, 16
        indices = make_subtask_indices(m=4, k=4)
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0]], batch_sizes=[100],
        )
        task_bits = x[:, m:]
        assert ((task_bits == 0) | (task_bits == 1)).all()

    def test_approximate_label_balance(self):
        """Labels should be approximately 50/50 for large batches."""
        m, n = 4, 16
        indices = make_subtask_indices(m=4, k=4)
        x, y = generate_cmsp_batch(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0]], batch_sizes=[10000],
        )
        fraction_ones = y.float().mean().item()
        assert 0.45 < fraction_ones < 0.55


class TestCMSPDataset:
    def test_length(self):
        m, k = 2, 3
        n = m * k
        indices = make_subtask_indices(m, k)
        ds = CMSPDataset(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0], [1]], samples_per_code=100,
        )
        assert len(ds) == 200

    def test_reproducibility(self):
        m, k = 2, 3
        n = m * k
        indices = make_subtask_indices(m, k)
        ds1 = CMSPDataset(n=n, m=m, subtask_indices=indices,
                          task_codes=[[0]], samples_per_code=100, seed=42)
        ds2 = CMSPDataset(n=n, m=m, subtask_indices=indices,
                          task_codes=[[0]], samples_per_code=100, seed=42)
        x1, y1 = ds1[0]
        x2, y2 = ds2[0]
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)

    def test_code_labels(self):
        m, k = 2, 3
        n = m * k
        indices = make_subtask_indices(m, k)
        ds = CMSPDataset(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0], [1]], samples_per_code=50,
        )
        assert (ds.code_labels[:50] == 0).all()
        assert (ds.code_labels[50:] == 1).all()


class TestDataloaders:
    def test_subtask_dataloaders(self):
        m, k = 2, 3
        n = m * k
        indices = make_subtask_indices(m, k)
        loaders = make_subtask_dataloaders(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0], [1]],
            samples_per_code=100, batch_size=32,
        )
        assert "{0}" in loaders
        assert "{1}" in loaders
        assert len(loaders) == 2

    def test_joint_dataloader(self):
        m, k = 2, 3
        n = m * k
        indices = make_subtask_indices(m, k)
        loader = make_joint_dataloader(
            n=n, m=m, subtask_indices=indices,
            task_codes=[[0], [1]],
            samples_per_code=100, batch_size=32,
        )
        total = sum(x.shape[0] for x, y in loader)
        assert total == 200

    def test_union_dataloaders(self):
        m, k = 2, 3
        n = m * k
        indices = make_subtask_indices(m, k)
        loaders = make_union_dataloaders(
            n=n, m=m, subtask_indices=indices,
            code_groups={"{0}∪{1}": [[0], [1]]},
            samples_per_code=100, batch_size=32,
        )
        assert "{0}∪{1}" in loaders
        # Union has 100 samples per code * 2 codes = 200
        total = sum(x.shape[0] for x, y in loaders["{0}∪{1}"])
        assert total == 200

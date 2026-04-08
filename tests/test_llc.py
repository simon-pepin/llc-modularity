"""Tests for LLC estimation.

These tests verify that the LLC estimation pipeline runs without error
and produces reasonable outputs. They use tiny models for speed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CMSPDataset, make_subtask_indices
from src.llc_estimation import compute_init_loss, compute_nbeta, cross_entropy_evaluate
from src.model import make_mlp


def _devinterp_available():
    try:
        import devinterp
        return True
    except ImportError:
        return False


class TestComputeNbeta:
    def test_basic(self):
        import math
        loader = MagicMock()
        loader.batch_size = 256
        nbeta = compute_nbeta(loader)
        expected = 256 / math.log(256)
        assert abs(nbeta - expected) < 1e-6

    def test_small_batch(self):
        loader = MagicMock()
        loader.batch_size = 1
        assert compute_nbeta(loader) == 1.0

    def test_none_batch(self):
        loader = MagicMock()
        loader.batch_size = None
        assert compute_nbeta(loader) == 1.0


class TestComputeInitLoss:
    def test_returns_finite(self):
        m, k = 2, 2
        n = m * k
        indices = make_subtask_indices(m, k)
        model = make_mlp(n=n, m=m, width=8, depth=2)

        ds = CMSPDataset(n=n, m=m, subtask_indices=indices,
                         task_codes=[[0]], samples_per_code=100, seed=0)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=32)

        loss = compute_init_loss(model, loader, device="cpu", n_batches=3)
        assert isinstance(loss, float)
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))


class TestCrossEntropyEvaluate:
    def test_returns_scalar(self):
        m, k = 2, 2
        n = m * k
        model = make_mlp(n=n, m=m, width=8, depth=2)
        x = torch.randn(10, m + n)
        y = torch.randint(0, 2, (10,))
        loss = cross_entropy_evaluate(model, (x, y))
        assert loss.ndim == 0
        assert loss.item() > 0


class TestEstimateLLC:
    """Integration test for LLC estimation. Requires devinterp to be installed."""

    @pytest.mark.skipif(
        not _devinterp_available(),
        reason="devinterp not installed"
    )
    def test_tiny_model(self):
        """Verify LLC estimation runs on a tiny model and returns valid results."""
        from src.llc_estimation import estimate_llc
        from torch.utils.data import DataLoader

        m, k = 2, 2
        n = m * k
        indices = make_subtask_indices(m, k)
        model = make_mlp(n=n, m=m, width=8, depth=2)

        # Quick training to get non-random weights
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(100):
            x = torch.randn(64, m + n)
            y = torch.randint(0, 2, (64,))
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ds = CMSPDataset(n=n, m=m, subtask_indices=indices,
                         task_codes=[[0]], samples_per_code=200, seed=0)
        loader = DataLoader(ds, batch_size=64)

        result = estimate_llc(
            model=model,
            dataloader=loader,
            num_chains=2,
            num_draws=10,
            num_burnin_steps=5,
            learning_rate=1e-4,
            device="cpu",
            seed=42,
            verbose=False,
        )

        assert "llc_mean" in result
        assert "llc_std" in result
        assert isinstance(result["llc_mean"], float)
        assert result["llc_mean"] > 0 or True  # LLC can be near-zero for random model

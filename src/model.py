"""MLP model definitions for CMSP experiments.

Adapted from ejmichaud/narrow. Standard ReLU MLPs with configurable
depth and width, matching the architectures used in the paper.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations.

    Architecture matches the narrow codebase: configurable depth and width,
    optional LayerNorm, 2-class output for binary classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 2,
        activation: str = "relu",
        use_layernorm: bool = False,
    ):
        """
        Args:
            input_dim: Input dimension (n_task_bits + n_control_bits).
            hidden_dims: List of hidden layer widths. Length = depth - 1.
            output_dim: Output dimension. Default 2 for binary classification
                with CrossEntropyLoss.
            activation: Activation function name ('relu', 'tanh', 'sigmoid').
            use_layernorm: Whether to add LayerNorm before each linear layer.
        """
        super().__init__()

        act_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }[activation]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            if use_layernorm:
                layers.append(nn.LayerNorm(prev_dim))
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            prev_dim = h_dim

        # Output layer
        if use_layernorm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_mlp(
    n: int,
    m: int,
    width: int = 128,
    depth: int = 3,
    activation: str = "relu",
    use_layernorm: bool = False,
) -> MLP:
    """Create an MLP with the standard CMSP configuration.

    Args:
        n: Number of task bits.
        m: Number of control bits (atomic subtasks).
        width: Hidden layer width.
        depth: Total depth (number of linear layers, including output).
        activation: Activation function name.
        use_layernorm: Whether to use LayerNorm.

    Returns:
        MLP model with input_dim = m + n, output_dim = 2.
    """
    input_dim = m + n
    hidden_dims = [width] * (depth - 1)
    return MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=2,
        activation=activation,
        use_layernorm=use_layernorm,
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

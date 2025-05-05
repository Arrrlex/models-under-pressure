from dataclasses import dataclass
import math
from typing import Any, Callable

import einops
import torch
import torch.nn as nn
from jaxtyping import Float


class BatchNorm(nn.Module):
    """
    Based on https://discuss.pytorch.org/t/how-does-the-batch-normalization-work-for-sequence-data/30839/2
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(embed_dim)

    def forward(
        self, activations: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    ) -> Float[torch.Tensor, "batch_size seq_len embed_dim"]:
        activations = activations.permute(0, 2, 1)
        activations = self.batch_norm(activations)
        activations = activations.permute(0, 2, 1)
        return activations


class Linear(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.linear = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self, activations: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    ) -> Float[torch.Tensor, "batch_size seq_len"]:
        return self.linear(activations).squeeze(-1)


class AttnLite(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, **kwargs: Any):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_query = Linear(embed_dim)

        self.classifier = Linear(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len embed_dim"],
        mask: Float[torch.Tensor, "batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]:
        attn_scores = self.context_query(x) / math.sqrt(self.embed_dim)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = einops.einsum(
            attn_weights, x, "batch seq, batch seq embed -> batch embed"
        )
        context = self.dropout(context)
        return self.classifier(context)


class LinearMeanPool(nn.Module):
    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__()
        self.batch_norm = BatchNorm(embed_dim)
        self.linear = Linear(embed_dim)

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len embed_dim"],
        mask: Float[torch.Tensor, "batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]:
        x = self.batch_norm(x)
        x = x.sum(dim=1) / mask.sum(dim=1, keepdims=True).clamp(min=1)
        return self.linear(x)


class LinearThenAgg(nn.Module):
    def __init__(self, embed_dim: int, agg: Callable, **kwargs: Any):
        super().__init__()
        self.batch_norm = BatchNorm(embed_dim)
        self.linear = Linear(embed_dim)
        self.agg = agg

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len embed_dim"],
        mask: Float[torch.Tensor, "batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]:
        x = self.batch_norm(x)
        x = self.linear(x)
        x = x.masked_fill(~mask, 0)
        x = self.agg(x, mask)
        return x


class LinearThenMean(LinearThenAgg):
    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__(embed_dim, mean_agg, **kwargs)


class LinearThenMax(LinearThenAgg):
    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__(embed_dim, max_agg, **kwargs)


class LinearThenSoftmax(LinearThenAgg):
    def __init__(self, embed_dim: int, temperature: float, **kwargs: Any):
        agg = SoftmaxAgg(temperature)
        super().__init__(embed_dim, agg, **kwargs)


class LinearThenRollingMax(LinearThenAgg):
    def __init__(self, embed_dim: int, window_size: int, **kwargs: Any):
        agg = RollingMaxAgg(window_size)
        super().__init__(embed_dim, agg, **kwargs)


class LinearThenLast(LinearThenAgg):
    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__(embed_dim, last_agg, **kwargs)


def mean_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def max_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.max(dim=1).values


@dataclass
class SoftmaxAgg:
    temperature: float

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # For softmax, mask with -inf
        x_for_softmax = x.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(x_for_softmax / self.temperature, dim=1)
        return (x * weights).sum(dim=1)


@dataclass
class RollingMaxAgg:
    window_size: int

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        windows = x.unfold(1, self.window_size, 1)
        window_means = windows.mean(dim=2)
        return window_means.max(dim=1).values


def last_agg(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[:, -1]

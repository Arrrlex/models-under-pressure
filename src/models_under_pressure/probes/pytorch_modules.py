import math
from typing import Callable

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from models_under_pressure.config import global_settings


class BatchNorm(nn.Module):
    """
    Based on https://discuss.pytorch.org/t/how-does-the-batch-normalization-work-for-sequence-data/30839/2
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.batch_norm = (
            nn.BatchNorm1d(embed_dim)
            .to(global_settings.DEVICE)
            .to(global_settings.DTYPE)
        )

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

        self.linear = (
            nn.Linear(embed_dim, 1, bias=False)
            .to(global_settings.DEVICE)
            .to(global_settings.DTYPE)
        )

    def forward(
        self, activations: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    ) -> Float[torch.Tensor, "batch_size seq_len"]:
        return self.linear(activations).squeeze(-1)


class AttnLite(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
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
    def __init__(self, embed_dim: int):
        super().__init__()
        self.batch_norm = BatchNorm(embed_dim)
        self.linear = Linear(embed_dim)

    def forward(
        self, x: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    ) -> Float[torch.Tensor, " batch_size"]:
        x = self.batch_norm(x)
        x = x.mean(dim=1)
        return self.linear(x)


class LinearThenAgg(nn.Module):
    def __init__(self, embed_dim: int, agg: Callable):
        super().__init__()
        self.batch_norm = BatchNorm(embed_dim)
        self.linear = Linear(embed_dim)
        self.agg = agg

    def forward(
        self, x: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    ) -> Float[torch.Tensor, " batch_size"]:
        x = self.batch_norm(x)
        x = self.linear(x)
        x = self.agg(x)
        return x


def mean_agg(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)


def max_agg(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=1).values


def mean_of_top_k(k: int) -> Callable:
    return lambda x: x.topk(k, dim=1).values.mean(dim=1)


def max_of_rolling_window(window_size: int) -> Callable:
    return lambda x: x.unfold(1, window_size, 1).max(dim=2).values

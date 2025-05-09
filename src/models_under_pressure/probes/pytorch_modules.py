import math
from typing import Any

import einops
import torch
import torch.nn as nn
from jaxtyping import Float


class AttnLite(nn.Module):
    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__()
        self.scale = math.sqrt(embed_dim)
        self.context_query = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len embed_dim"],
        mask: Float[torch.Tensor, "batch_size seq_len"],
        return_per_token: bool = False,
    ) -> (
        Float[torch.Tensor, " batch_size"]
        | tuple[
            Float[torch.Tensor, " batch_size"],
            Float[torch.Tensor, " batch_size seq_len"],
            Float[torch.Tensor, " batch_size seq_len"],
        ]
    ):
        attn_scores = self.context_query(x).squeeze(-1) / self.scale
        masked_attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(masked_attn_scores, dim=-1)
        token_logits = self.classifier(x).squeeze(-1)
        token_logits = token_logits.masked_fill(~mask, 0)

        context = einops.einsum(
            attn_weights, x, "batch seq, batch seq embed -> batch embed"
        )
        sequence_logits = self.classifier(context).squeeze(-1)

        if return_per_token:
            return sequence_logits, attn_scores, token_logits
        else:
            return sequence_logits


class MeanThenLinear(nn.Module):
    def __init__(self, embed_dim: int, **kwargs: Any):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len embed_dim"],
        mask: Float[torch.Tensor, "batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]:
        x = x.sum(dim=1) / mask.sum(dim=1, keepdims=True).clamp(min=1)
        return self.linear(x).squeeze(-1)


class LinearThenAgg(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        **kwargs: Any,
    ):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        self.kwargs = kwargs

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size seq_len embed_dim"],
        mask: Float[torch.Tensor, "batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]:
        x = self.linear(x).squeeze(-1)
        x = x.masked_fill(~mask, 0)
        x = self.agg(x, mask)
        return x

    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")


class LinearThenMean(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class LinearThenMax(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values


class LinearThenSoftmax(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        temperature = self.kwargs["temperature"]
        x_for_softmax = x.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(x_for_softmax / temperature, dim=1)
        return (x * weights).sum(dim=1)


class LinearThenRollingMax(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        window_size = self.kwargs["window_size"]
        windows = x.unfold(1, window_size, 1)
        window_means = windows.mean(dim=2)
        return window_means.max(dim=1).values


class LinearThenLast(LinearThenAgg):
    def agg(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        idxs = (
            torch.arange(mask.size(1), device=mask.device).unsqueeze(0).expand_as(mask)
        )
        # Set invalid positions to -1, valid to their index
        idxs = idxs * mask + (~mask) * -1
        last_valid = idxs.max(dim=1).values  # (batch,)
        batch_indices = torch.arange(x.size(0), device=x.device)
        return x[batch_indices, last_valid]

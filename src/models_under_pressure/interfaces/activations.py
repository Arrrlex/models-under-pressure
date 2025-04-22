from dataclasses import dataclass

import einops
import torch
from jaxtyping import Float
from torch.utils.data import Dataset as TorchDataset

from models_under_pressure.config import global_settings
from models_under_pressure.interfaces.dataset import BaseDataset


class ActivationDataset(TorchDataset):
    """
    A pytorch Dataset class that contains the activations structured as a batch-wise dataset.
    Each activation and attention mask is batch_size, seq_len, (embed_dim).
    """

    def __init__(
        self,
        activations: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        y: torch.Tensor,
    ):
        self.activations = activations
        self.attention_mask = attention_mask
        self.input_ids = input_ids
        self.y = y

        self.set_dtype = self.activations.dtype != global_settings.DTYPE
        self.set_device = self.activations.device != global_settings.DEVICE

        # We've already put y on the correct device and dtype
        assert y.device == global_settings.DEVICE
        assert y.dtype == global_settings.DTYPE

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the masked activations, attention mask, input ids and label.
        """

        act = self.activations[index]
        attn = self.attention_mask[index]
        ids = self.input_ids[index]
        y = self.y[index]

        if self.set_device:
            act = act.to(global_settings.DEVICE)
            attn = attn.to(global_settings.DEVICE)
            ids = ids.to(global_settings.DEVICE)

        if self.set_dtype:
            act = act.to(global_settings.DTYPE)
            attn = attn.to(global_settings.DTYPE)
            ids = ids.to(global_settings.DTYPE)

        return act, attn, ids, y


@dataclass
class Activation:
    activations: Float[torch.Tensor, "batch_size seq_len embed_dim"]
    attention_mask: Float[torch.Tensor, "batch_size seq_len"]
    input_ids: Float[torch.Tensor, "batch_size seq_len"]

    @classmethod
    def from_dataset(cls, dataset: BaseDataset) -> "Activation":
        return cls(
            activations=dataset.other_fields["activations"],  # type: ignore
            attention_mask=dataset.other_fields["attention_mask"],  # type: ignore
            input_ids=dataset.other_fields["input_ids"],  # type: ignore
        )

    def __post_init__(self):
        """Validate shapes after initialization, applies attention mask to activations."""
        shape = (self.batch_size, self.seq_len)
        assert (
            self.attention_mask.shape == shape
        ), f"Attention mask shape {self.attention_mask.shape} doesn't agree with {shape}"
        assert (
            self.input_ids.shape == shape
        ), f"Input ids shape {self.input_ids.shape} doesn't agree with {shape}"

        self.activations *= self.attention_mask[:, :, None]
        self.shape = self.activations.shape
        self.batch_size, self.seq_len, self.embed_dim = self.shape

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "Activation":
        return Activation(
            activations=self.activations.to(device).to(dtype),
            attention_mask=self.attention_mask.to(device).to(dtype),
            input_ids=self.input_ids.to(device).to(dtype),
        )

    def per_token(self) -> "PerTokenActivation":
        activations = einops.rearrange(self.activations, "b s e -> (b s) e")
        attention_mask = einops.rearrange(self.attention_mask, "b s -> (b s)")
        input_ids = einops.rearrange(self.input_ids, "b s -> (b s)")
        return PerTokenActivation(
            activations=activations, attention_mask=attention_mask, input_ids=input_ids
        )

    def to_dataset(self, y: Float[torch.Tensor, " batch_size"]) -> ActivationDataset:
        return ActivationDataset(
            activations=self.activations,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            y=y,
        )


@dataclass
class PerTokenActivation:
    activations: Float[torch.Tensor, "tokens embed_dim"]
    attention_mask: Float[torch.Tensor, " tokens"]
    input_ids: Float[torch.Tensor, " tokens"]

    def to_dataset(self, y: Float[torch.Tensor, " batch_size"]) -> ActivationDataset:
        tokens = self.activations.shape[0]
        seq_len, rem = divmod(tokens, y.shape[0])
        assert (
            rem == 0
        ), f"Batch size {y.shape[0]} does not divide the number of tokens {tokens}"
        y = einops.repeat(y, "b -> (b s)", s=seq_len)
        return ActivationDataset(
            activations=self.activations,
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            y=y,
        )

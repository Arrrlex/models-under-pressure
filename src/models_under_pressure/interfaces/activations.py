from dataclasses import dataclass

import einops
import torch
from jaxtyping import Float
from torch.utils.data import Dataset as TorchDataset

from models_under_pressure.config import global_settings
from models_under_pressure.interfaces.dataset import BaseDataset


class ActivationPerTokenDataset(TorchDataset):
    """
    A pytorch Dataset class that contains the activations structured as a flattened per-token dataset.
    Each activation and attention mask is batch_size * seq_len long.
    """

    def __init__(
        self,
        activations: "Activation",
        y: Float[torch.Tensor, " batch_size"],
    ):
        self.activations = einops.rearrange(activations.activations, "b s e -> (b s) e")
        self.attention_mask = einops.rearrange(
            activations.attention_mask, "b s -> (b s)"
        )
        self.input_ids = einops.rearrange(activations.input_ids, "b s -> (b s)")
        self.y = einops.repeat(y, "b -> (b s)", s=activations.seq_len)

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[
        Float[torch.Tensor, " embed_dim"],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        """
        Returns the masked activations, attention mask, input id and label.
        """

        # return (
        #     self.activations[index].to(global_settings.DEVICE),
        #     self.attention_mask[index].to(global_settings.DEVICE),
        #     self.input_ids[index].to(global_settings.DEVICE),
        #     self.y[index].to(global_settings.DEVICE),
        # )
        return (
            self.activations[index],
            self.attention_mask[index],
            self.input_ids[index],
            self.y[index],
        )


class ActivationDataset(TorchDataset):
    """
    A pytorch Dataset class that contains the activations structured as a batch-wise dataset.
    Each activation and attention mask is batch_size, seq_len, (embed_dim).
    """

    def __init__(
        self,
        activations: "Activation",
        y: Float[torch.Tensor, " batch_size"],
    ):
        self.activations = activations.activations
        self.attention_mask = activations.attention_mask
        self.input_ids = activations.input_ids
        self.y = y

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[
        Float[torch.Tensor, "seq_len embed_dim"],
        Float[torch.Tensor, " seq_len"],
        Float[torch.Tensor, " seq_len"],
        Float[torch.Tensor, ""],
    ]:
        """
        Return the masked activations, attention mask, input ids and label.
        """

        return (
            self.activations[index].to(global_settings.DEVICE),
            self.attention_mask[index].to(global_settings.DEVICE),
            self.input_ids[index].to(global_settings.DEVICE),
            self.y[index].to(global_settings.DEVICE),
        )


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

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.activations.shape  # type: ignore

    @property
    def batch_size(self) -> int:
        return self.shape[0]  # type: ignore

    @property
    def seq_len(self) -> int:
        return self.shape[1]  # type: ignore

    @property
    def embed_dim(self) -> int:
        return self.shape[2]  # type: ignore

    def __post_init__(self):
        """Validate shapes after initialization."""
        shape = (self.batch_size, self.seq_len)
        assert (
            self.attention_mask.shape == shape
        ), f"Attention mask shape {self.attention_mask.shape} doesn't agree with {shape}"
        assert (
            self.input_ids.shape == shape
        ), f"Input ids shape {self.input_ids.shape} doesn't agree with {shape}"

        self.activations *= self.attention_mask[:, :, None]

    def to_dataset(
        self, y: Float[torch.Tensor, " batch_size"], per_token: bool = False
    ) -> "ActivationDataset | ActivationPerTokenDataset":
        dataset_type = ActivationPerTokenDataset if per_token else ActivationDataset
        return dataset_type(activations=self, y=y)

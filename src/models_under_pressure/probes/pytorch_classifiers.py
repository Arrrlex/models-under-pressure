from dataclasses import dataclass, field

import einops
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models_under_pressure.config import PYTORCH_PT_TRAINING_ARGS
from models_under_pressure.interfaces.activations import (
    Activation,
)
from typing import Callable
import functools


@dataclass
class PytorchLinearClassifier:
    """
    A linear classifier that uses PyTorch. The model trains on the flattened batch_size * seq_len
    activations and labels.
    """

    model: nn.Module | None = None
    training_args: dict = field(default_factory=lambda: PYTORCH_PT_TRAINING_ARGS)

    def train(self, activations: Activation, y: Float[np.ndarray, " batch_size"]):
        """
        Train the classifier on the activations and labels.

        Args:
            activations: The activations to train on.
            y: The labels to train on.

        Returns:
            None - The self.model is updated in place!
        """

        device = self.training_args["device"]

        # Create a linear model
        if self.model is None:
            self.model = self.create_model(activations.shape[2]).to(device)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        per_token_dataset = activations.to_dataset(y=y, per_token=True)

        # Calculate class weights to handle imbalanced data
        sample_weights = per_token_dataset._attention_mask

        # Create weighted sampler
        # Only sample points that are not masked
        sampler = WeightedRandomSampler(
            weights=sample_weights.numpy(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        dataloader = DataLoader(
            per_token_dataset,
            batch_size=self.training_args["batch_size"],
            sampler=sampler,
        )

        # Training loop
        self.model.train()
        for epoch in range(self.training_args["epochs"]):
            running_loss = 0.0
            pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.training_args['epochs']}"
            )
            for batch_idx, batch in enumerate(pbar):
                acts, _, _, y = batch

                # Preprocessing step on the activations:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(acts.to(device))
                loss = criterion(outputs.squeeze(), y.to(device))  # type: ignore

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update running loss and progress bar
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Print epoch summary
            print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        return self

    def predict(self, activations: Activation) -> Float[np.ndarray, " batch_size"]:
        """
        Predict the labels of the activations.
        """

        # Get the probabilities
        probs = self.predict_proba(activations)

        # Take the mean over the sequence length:
        probs = probs.mean(axis=1)

        # Get the predictions -> cutoff at 0.5
        preds = (probs > 0.5).to(torch.int32)  # type: ignore

        # Convert the predictions to a numpy array
        return preds.numpy()

    @torch.no_grad()
    def predict_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size seq_len"]:
        """
        Predict the probabilities of the activations.

        Outputs are expected in the shape (batch_size,)
        """

        logits = self.predict_token_logits(activations)

        # Take the mean over the sequence length:
        mean_logits = logits.mean(axis=1)

        # Convert the logits to probabilities
        return torch.sigmoid(mean_logits).numpy()

    @torch.no_grad()
    def predict_token_logits(
        self, activations: Activation
    ) -> Float[torch.Tensor, " batch_size seq_len"]:
        """
        Predict the logits of the activations.

        Outputs are expected in the shape (batch_size, seq_len)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        device = self.training_args["device"]

        # Process the activations into a per token dataset to be passed through the model
        batch_size, seq_len, embed_dim = activations.shape

        acts_tensor = activations.get_activations(per_token=True)

        # Switch batch norm to eval mode:
        self.model = self.model.to(device)
        self.model.eval()

        logits = self.model(torch.tensor(acts_tensor, dtype=torch.float32).to(device))

        # Multiply by the attention mask -> to remove padded tokens:
        masked_logits = logits * torch.tensor(
            activations.get_attention_mask(per_token=True)[:, None], dtype=torch.float32
        ).to(device)

        # Reshape back to the original shape and take the mean over the sequence length
        reshaped_logits = einops.rearrange(
            masked_logits, "(b s) 1 -> b s", b=batch_size, s=seq_len
        )

        assert reshaped_logits.shape == (
            batch_size,
            seq_len,
        ), f"Logits shape is {reshaped_logits.shape} not {(batch_size, seq_len)}"

        return reshaped_logits

    @torch.no_grad()
    def predict_token_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size seq_len"]:
        """
        Predict the probabilities of the activations.

        Outputs are expected in the shape (batch_size, seq_len)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Process the activations into a per token dataset to be passed through the model
        logits = self.predict_token_logits(activations)

        # Convert the logits to probabilities
        return torch.sigmoid(logits).numpy()

    def create_model(self, embedding_dim: int) -> nn.Module:
        """
        Create a linear model over the embedding dimension dynamically.
        """

        # Create a linear model over the embedding dimension
        return nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False),
        )


@dataclass
class PytorchAttentionClassifier:
    """
    A linear classifier that uses PyTorch. The sequence is aggregated using a learnt attention mechanism.
    """

    model: nn.Module | None = None
    training_args: dict = field(default_factory=lambda: PYTORCH_PT_TRAINING_ARGS)


def mean_score(
    x: Float[np.ndarray, " batch_size seq_len"],
) -> Float[np.ndarray, " batch_size"]:
    return x.mean(axis=1)


def max_score(
    x: Float[np.ndarray, " batch_size seq_len"],
) -> Float[np.ndarray, " batch_size"]:
    return x.max(axis=1)


def max_of_rolling_mean(
    window_size: int,
) -> Callable[
    [Float[torch.Tensor, " batch_size seq_len"]], Float[torch.Tensor, " batch_size"]
]:
    def max_of_rolling_mean_impl(
        x: Float[torch.Tensor, " batch_size seq_len"],
    ) -> Float[torch.Tensor, " batch_size"]:
        return (
            x.unfold(dimension=1, size=window_size, step=1)
            .mean(dim=2)
            .max(dim=1)
            .values
        )

    return max_of_rolling_mean_impl


@dataclass
class PytorchAggregationClassifier(PytorchLinearClassifier):
    """
    A linear classifier that uses PyTorch. The sequence is aggregated using a custom aggregation function.
    """

    aggregation_function: Callable = mean_score
    before_sigmoid: bool = False

    def predict_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size"]:
        """
        Predict the probabilities of the activations.
        """
        if self.before_sigmoid:
            functions = [
                self.predict_token_logits,
                torch.sigmoid,
                self.aggregation_function,
            ]
        else:
            functions = [
                self.predict_token_logits,
                self.aggregation_function,
                torch.sigmoid,
            ]

        return functools.reduce(lambda x, f: f(x), functions, activations)

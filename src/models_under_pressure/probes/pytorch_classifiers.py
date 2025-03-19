from dataclasses import dataclass, field

import einops
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models_under_pressure.interfaces.activations import Activation


@dataclass
class PytorchLinearClassifier:
    """
    A linear classifier that uses PyTorch. The model trains on the flattened batch_size * seq_len
    activations and labels.
    """

    model: nn.Module | None = None
    training_args: dict = field(
        default_factory=lambda: {
            "batch_size": 16,
            "epochs": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
        }
    )

    def train(self, activations: Activation, y: Float[np.ndarray, " batch_size"]):
        """
        Train the classifier on the activations and labels.

        Args:
            activations: The activations to train on.
            y: The labels to train on.

        Returns:
            None - The self.model is updated in place!
        """

        # Create a linear model
        if self.model is None:
            self.model = self.create_model(activations.shape)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters())
        criterion = nn.BCEWithLogitsLoss()

        dataset = activations.to_dataset(y=y)
        per_token_dataset = dataset.to_per_token()

        # Calculate class weights to handle imbalanced data
        sample_weights = per_token_dataset.attention_mask

        # Create weighted sampler
        # Only sample points that are not masked
        sampler = WeightedRandomSampler(
            weights=sample_weights.numpy(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        dataloader = DataLoader(per_token_dataset, batch_size=16, sampler=sampler)

        # Training loop
        self.model.train()
        for _ in tqdm(range(self.training_args["epochs"]), desc="Training epochs"):
            for batch in dataloader:
                acts, _, _, y = batch

                # Preprocessing step on the activations:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(acts)
                loss = criterion(outputs.squeeze(), y)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

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
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Process the activations into a per token dataset to be passed through the model
        batch_size, seq_len, embed_dim = activations.shape

        activations = activations.to_per_token()

        # Get the logits
        logits = self.model(torch.tensor(activations.activations))

        # Reshape back to the original shape and take the mean over the sequence length
        logits = einops.rearrange(logits, "(b s) 1 -> b s", b=batch_size, s=seq_len)

        assert logits.shape == (
            batch_size,
            seq_len,
        ), f"Logits shape is {logits.shape} not {(batch_size, seq_len)}"

        # Convert the logits to probabilities
        return torch.sigmoid(logits).numpy()

    def create_model(self, activations_shape: tuple[int, int, int]) -> nn.Module:
        """
        Create a linear model over the embedding dimension dynamically.
        """

        # Create a linear model over the embedding dimension
        return nn.Linear(activations_shape[2], 1)

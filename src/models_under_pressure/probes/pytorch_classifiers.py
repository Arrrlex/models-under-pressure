from dataclasses import dataclass
from typing import Self

import einops
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models_under_pressure.interfaces.activations import (
    Activation,
)


@dataclass
class PytorchLinearClassifier:
    """
    A linear classifier that uses PyTorch. The model trains on the flattened batch_size * seq_len
    activations and labels.
    """

    training_args: dict
    model: nn.Module | None = None

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
        # optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.training_args.get("learning_rate", 1e-3),
        #     weight_decay=self.training_args.get("weight_decay", 0.01),
        # )

        # TODO We probably want to switch back to AdamW here, or allow for both optimizers
        optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=100)

        criterion = nn.BCEWithLogitsLoss()

        per_token_dataset = activations.to_dataset(y=y, per_token=True)

        # Calculate class weights to handle imbalanced data
        sample_weights = per_token_dataset._attention_mask

        # Create weighted sampler
        # Only sample points that are not masked
        sampler = WeightedRandomSampler(
            weights=sample_weights.numpy(),  # type: ignore
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
                # TODO! Adjust this so it applies the mask correctly (as for per-entry classifier)
                acts, _, _, y = batch

                # Define closure for LBFGS
                def closure():
                    optimizer.zero_grad()
                    outputs = self.model(acts.to(device))
                    loss = criterion(outputs.squeeze(), y.to(device))  # type: ignore
                    loss.backward()
                    return loss

                # Optimize using the closure
                loss = optimizer.step(closure)

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
        # probs = probs.mean(axis=1)

        # Get the predictions -> cutoff at 0.5
        preds = (probs > 0.5).astype(np.int32)

        # Convert the predictions to a numpy array
        return preds

    @torch.no_grad()
    def predict_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size"]:
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
    ) -> Float[np.ndarray, " batch_size seq_len"]:
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
        # return torch.sigmoid(torch.tensor(logits)).numpy()
        return torch.tensor(logits).numpy()

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

    training_args: dict
    model: nn.Module | None = None


@dataclass
class PytorchDifferenceOfMeansClassifier(PytorchLinearClassifier):
    use_lda: bool = False

    def train(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ) -> Self:
        acts = torch.tensor(activations.get_activations(), dtype=torch.float32)
        mask = torch.tensor(activations.get_attention_mask(), dtype=torch.float32)

        batch_size, seq_len, embed_dim = acts.shape

        acts = acts.to(self.training_args["device"])
        mask = mask.to(self.training_args["device"])
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.training_args["device"])

        # Apply mask to zero out irrelevant entries
        masked_acts = acts * mask.unsqueeze(
            -1
        )  # broadcast mask across embedding dimension

        # Sum along sequence length and divide by mask sum for each sample
        mask_sums = mask.sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
        averaged_acts = (
            masked_acts.sum(dim=1) / mask_sums
        )  # shape: (batch_size, embed_dim)

        # Separate positive and negative examples
        pos_acts = averaged_acts[y_tensor == 1]
        neg_acts = averaged_acts[y_tensor == 0]

        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        if self.use_lda:
            centered_data = torch.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
            covariance = centered_data.t() @ centered_data / acts.shape[0]

            inv = torch.linalg.pinv(covariance, hermitian=True, atol=1e-3)
            param = inv @ direction
        else:
            param = direction

        assert param.shape == (embed_dim,)

        self.model = nn.Linear(embed_dim, 1, bias=False)
        self.model.weight.data.copy_(param.reshape(1, -1))

        # Set bias to put decision boundary halfway between positive and negative means
        # TODO Unsure if this is fine for unbalanced datasets
        # TODO Setting a bias leads to much lower AUROC, which doesn't really make sense
        # pos_logits = (pos_mean @ param).mean()
        # neg_logits = (neg_mean @ param).mean()
        # bias = -(pos_logits + neg_logits) / 2
        # self.model.bias.data.copy_(bias.reshape(1))

        return self


def average_activations(acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_acts = acts * mask[..., None]
    token_counts = mask.sum(dim=1, keepdim=True)
    # Add small epsilon to avoid division by zero
    token_counts = torch.clamp(token_counts, min=1)
    averaged_acts = masked_acts.sum(dim=1) / token_counts
    return averaged_acts


class PytorchPerEntryLinearClassifier(PytorchLinearClassifier):
    """
    A linear classifier that uses Pytorch with standardization of features.
    """

    training_args: dict
    model: nn.Module | None = None

    def train(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ):
        device = self.training_args["device"]
        optimizer_type = self.training_args["optimizer_type"]
        weight_decay = self.training_args.get("weight_decay", 0.01)
        learning_rate = self.training_args.get("learning_rate", 1e-3)

        # Just this bit here is different from the PytorchLinearClassifier
        per_entry_dataset = activations.to_dataset(y=y, per_token=False)

        # Process the dataset to be the mean across the seq_len:
        acts = per_entry_dataset._activations
        mask = per_entry_dataset._attention_mask

        averaged_acts = average_activations(acts, mask)

        # Get the embedding dimension from the averaged activations
        embedding_dim = averaged_acts.shape[-1]
        # Create a linear model with the correct embedding dimension
        if self.model is None:
            self.model = self.create_model(embedding_dim).to(device)
            # Initialize the linear layer weights to zeros
            if isinstance(self.model, nn.Sequential):
                for i, module in enumerate(self.model):
                    if isinstance(module, nn.Linear):
                        # torch.nn.init.zeros_(module.weight)
                        torch.nn.init.xavier_uniform_(module.weight)

        # Ensure model is not None
        if self.model is None:
            raise ValueError("Failed to create model")

        # Filter out learning_rate, epochs and batch_size from training args
        optimizer_args = {
            k: v
            for k, v in self.training_args.items()
            if k
            not in [
                "learning_rate",
                "epochs",
                "batch_size",
                "device",
                "weight_decay",
                "optimizer_type",
            ]
        }
        # Choose optimizer based on the optimizer_type parameter
        if optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_args,
            )
        else:  # Default to LBFGS
            optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=learning_rate, **optimizer_args
            )

        criterion = nn.BCEWithLogitsLoss()

        dataloader = DataLoader(
            per_entry_dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=True,
        )

        # Training loop
        self.model.train()
        for epoch in range(self.training_args["epochs"]):
            running_loss = 0.0
            pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.training_args['epochs']}"
            )
            for batch_idx, batch in enumerate(pbar):
                acts, mask_batch, _, y = batch
                # acts_tensor = torch.tensor(acts, dtype=torch.float32).to(device)
                # mask_tensor = torch.tensor(mask_batch, dtype=torch.float32).to(device)
                acts_tensor = average_activations(acts, mask_batch)
                y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

                if optimizer_type.lower() == "adamw":
                    # Standard training step for AdamW
                    optimizer.zero_grad()
                    outputs = self.model(acts_tensor)
                    loss = criterion(outputs.squeeze(), y_tensor)
                    loss.backward()
                    optimizer.step()
                    loss = loss.item()
                else:
                    # Define closure for LBFGS with regularization
                    def closure():
                        optimizer.zero_grad()
                        outputs = self.model(acts_tensor)
                        bce_loss = criterion(outputs.squeeze(), y_tensor)

                        # Simpler L2 regularization
                        l2_reg = 0.0
                        for param in self.model.parameters():
                            l2_reg += 0.5 * weight_decay * (param**2).sum()

                        total_loss = bce_loss + l2_reg
                        total_loss.backward()
                        return total_loss

                    # Optimize using the closure
                    loss = optimizer.step(closure)
                    if loss is None:
                        loss = 0.0  # Handle case where LBFGS step returns None

                # Update running loss and progress bar
                running_loss += loss
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
        # probs = probs.mean(axis=1)

        # Get the predictions -> cutoff at 0.5
        preds = (probs > 0.5).astype(np.int32)

        # Convert the predictions to a numpy array
        return preds

    @torch.no_grad()
    def predict_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size"]:
        """
        Predict the probabilities of the activations.

        Outputs are expected in the shape (batch_size,)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        device = self.training_args["device"]

        # Process the activations into a per token dataset to be passed through the model
        batch_size, seq_len, embed_dim = activations.shape

        # Switch batch norm to eval mode:
        self.model = self.model.to(device)
        self.model.eval()

        acts_tensor = average_activations(
            torch.tensor(activations._activations, dtype=torch.float32),
            torch.tensor(activations._attention_mask, dtype=torch.float32),
        )
        logits = self.model(acts_tensor.to(device))

        # Multiply by the attention mask -> to remove padded tokens:
        # masked_logits = logits * torch.tensor(
        #    activations.get_attention_mask(per_token=True)[:, None], dtype=torch.float32
        # ).to(device)

        # Reshape back to the original shape and take the mean over the sequence length
        # reshaped_logits = einops.rearrange(
        #    masked_logits, "(b s) 1 -> b s", b=batch_size, s=seq_len
        # )

        return logits.squeeze().numpy()

    @torch.no_grad()
    def predict_token_logits(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size seq_len"]:
        """
        Predict the logits of the activations.

        Outputs are expected in the shape (batch_size, seq_len)
        """
        raise NotImplementedError("Not implemented")

    @torch.no_grad()
    def predict_token_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size seq_len"]:
        """
        Predict the probabilities of the activations.

        Outputs are expected in the shape (batch_size, seq_len)
        """
        raise NotImplementedError("Not implemented")

    def create_model(self, embedding_dim: int) -> nn.Module:
        """
        Create a linear model over the embedding dimension dynamically.
        """

        # Create a linear model over the embedding dimension
        return nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False),
            nn.Sigmoid(),
        )

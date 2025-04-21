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
    data_type: torch.dtype | None = None
    best_epoch: int | None = None

    def __post_init__(self):
        # Set data type based on available device
        device = self.training_args.get("device", "auto")
        if torch.cuda.is_available() and ("cuda" in device or "auto" in device):
            self.data_type = torch.bfloat16
        else:
            self.data_type = torch.float32

    def train(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[np.ndarray, " batch_size"] | None = None,
        print_gradient_norm: bool = False,
    ) -> Self:
        """
        Train the classifier on the activations and labels.

        Args:
            activations: The activations to train on.
            y: The labels to train on.
            validation_activations: Optional validation activations.
            validation_labels: Optional validation labels.

        Returns:
            Self - The trained classifier.
        """
        device = self.training_args["device"]

        # Create a linear model
        if self.model is None:
            self.model = self.create_model(activations.shape[2]).to(device)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.training_args["optimizer_args"],
        )

        criterion = nn.BCEWithLogitsLoss()

        per_token_dataset = activations.to_dataset(
            y=torch.tensor(y, dtype=self.data_type), per_token=True
        )

        # Calculate class weights to handle imbalanced data
        sample_weights = per_token_dataset._attention_mask

        # Only sample points that are not masked
        sampler = WeightedRandomSampler(
            weights=sample_weights.numpy().tolist(),  # Convert to list for compatibility
            num_samples=len(sample_weights),
            replacement=True,
        )

        dataloader = DataLoader(
            per_token_dataset,
            batch_size=self.training_args["batch_size"],
            sampler=sampler,
        )

        # Initialize variables for tracking best model
        best_val_loss = float("inf")
        best_model_state = None

        # Training loop
        for epoch in range(self.training_args["epochs"]):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.training_args['epochs']}"
            )
            for batch_idx, batch in enumerate(pbar):
                acts_tensor, _, _, y_tensor = batch

                optimizer.zero_grad()
                outputs = self.model(acts_tensor.to(device).to(self.data_type))
                loss = criterion(outputs.squeeze(), y_tensor.to(device))
                loss.backward()

                if print_gradient_norm:
                    # Calculate and print gradient norm
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    print(f"gradient norm: {total_norm}")

                optimizer.step()
                loss = loss.item()

                # Update running loss and progress bar
                running_loss += loss
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Print epoch summary
            print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

            # Validation step if validation data is provided
            if validation_activations is not None and validation_y is not None:
                self.model.eval()
                with torch.no_grad():
                    # Get probabilities for validation data
                    val_probs = self.predict_proba(validation_activations)

                    # Convert validation labels to tensor
                    val_y_tensor = torch.tensor(validation_y, dtype=self.data_type).to(
                        device
                    )

                    # Convert probabilities to logits and compute loss
                    val_probs_tensor = torch.tensor(val_probs, dtype=self.data_type).to(
                        device
                    )

                    # Convert to logits first, then handle extreme values
                    val_logits = torch.logit(val_probs_tensor)

                    # Clip extreme logit values to prevent NaN in loss computation
                    # Using values that are safe for bfloat16
                    val_logits = torch.clamp(val_logits, min=-10.0, max=10.0)

                    # Check for NaN values in logits
                    if torch.isnan(val_logits).any():
                        print("Warning: NaN values detected in validation logits")
                        print("Min logit:", val_logits.min().item())
                        print("Max logit:", val_logits.max().item())
                        val_logits = torch.nan_to_num(val_logits, nan=0.0)

                    val_loss = criterion(val_logits.squeeze(), val_y_tensor).item()

                    # Check for NaN loss
                    if np.isnan(val_loss):
                        print("Warning: NaN validation loss detected")
                        print("Validation probabilities shape:", val_probs_tensor.shape)
                        print("Validation labels shape:", val_y_tensor.shape)
                        val_loss = float(
                            "inf"
                        )  # Set to infinity to avoid selecting this model

                    print(f"Validation loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.model.state_dict().copy()
                        self.best_epoch = epoch + 1  # Store 1-indexed epoch number

        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self

    @torch.no_grad()
    def predict_token_logits(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size seq_len"]:
        """
        Predict the logits of the activations.

        1. Get the activations
        2. Pass through the model
        3. Multiply by the attention mask
        4. Reshape back to (batch_size, seq_len) and then return.

        Outputs the classifier logits that are expected in the shape (batch_size, seq_len)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        device = self.training_args["device"]

        # Process the activations into a per token dataset to be passed through the model
        batch_size, seq_len, _ = activations.shape

        acts_tensor = activations.get_activations(per_token=True)

        # Switch batch norm to eval mode:
        self.model = self.model.to(device)
        self.model.eval()

        logits = self.model(torch.tensor(acts_tensor, dtype=self.data_type).to(device))

        # Multiply by the attention mask -> to remove padded tokens:
        masked_logits = logits * torch.tensor(
            activations.get_attention_mask(per_token=True)[:, None],
            dtype=self.data_type,
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

    def predict(self, activations: Activation) -> Float[np.ndarray, " batch_size"]:
        """
        Predict the per entry labels of the inputs.
        """

        # Get the probabilities
        probs = self.predict_proba(activations)

        # Get the predictions -> cutoff at 0.5
        preds = (probs > 0.5).astype(np.int32)

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
        return torch.sigmoid(mean_logits).cpu().float().numpy()

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
        return torch.sigmoid(torch.tensor(logits)).numpy()

    def create_model(self, embedding_dim: int) -> nn.Module:
        """
        Create a linear model over the embedding dimension dynamically.
        """

        # Create a linear model over the embedding dimension
        return nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False),
        ).to(self.data_type)


@dataclass
class PytorchDifferenceOfMeansClassifier(PytorchLinearClassifier):
    use_lda: bool = False

    def train(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
    ) -> Self:
        acts = torch.tensor(activations.get_activations(), dtype=self.data_type)
        mask = torch.tensor(activations.get_attention_mask(), dtype=self.data_type)

        batch_size, seq_len, embed_dim = acts.shape

        acts = acts.to(self.training_args["device"])
        mask = mask.to(self.training_args["device"])
        y_tensor = torch.tensor(y, dtype=self.data_type).to(
            self.training_args["device"]
        )

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
        validation_activations: Activation | None = None,
        validation_y: Float[np.ndarray, " batch_size"] | None = None,
        print_gradient_norm: bool = False,
    ) -> Self:
        """
        Train the classifier on the activations and labels.

        Args:
            activations: The activations to train on.
            y: The labels to train on.
            validation_activations: Optional validation activations.
            validation_y: Optional validation labels.
            print_gradient_norm: Whether to print gradient norm during training.

        Returns:
            Self - The trained classifier.
        """
        device = self.training_args["device"]

        # Just this bit here is different from the PytorchLinearClassifier
        per_entry_dataset = activations.to_dataset(
            y=torch.tensor(y, dtype=self.data_type), per_token=False
        )

        # Get the embedding dimension from the averaged activations
        embedding_dim = activations._activations.shape[-1]
        # Create a linear model with the correct embedding dimension

        if self.model is None:
            self.model = self.create_model(embedding_dim).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.training_args["optimizer_args"],
        )

        criterion = nn.BCEWithLogitsLoss()

        dataloader = DataLoader(
            per_entry_dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=True,
        )

        # Initialize variables for tracking best model
        best_val_loss = float("inf")
        best_model_state = None

        # Training loop
        self.model.train()
        for epoch in range(self.training_args["epochs"]):
            running_loss = 0.0
            pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{self.training_args['epochs']}"
            )
            for batch_idx, batch in enumerate(pbar):
                acts, mask_batch, _, y = batch

                acts_tensor = self.average_activations(acts, mask_batch)
                y_tensor = torch.tensor(y, dtype=self.data_type).to(device)

                # Standard training step for AdamW
                optimizer.zero_grad()
                outputs = self.model(acts_tensor.to(self.data_type).to(device))
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()

                if print_gradient_norm:
                    # Calculate and print gradient norm
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    print(f"gradient norm: {total_norm}")

                optimizer.step()
                loss = loss.item()

                # Update running loss and progress bar
                running_loss += loss
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Print epoch summary
            print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

            # Validation step if validation data is provided
            if validation_activations is not None and validation_y is not None:
                self.model.eval()
                with torch.no_grad():
                    # Get probabilities for validation data
                    val_probs = self.predict_proba(validation_activations)

                    # Convert validation labels to tensor
                    val_y_tensor = torch.tensor(validation_y, dtype=self.data_type).to(
                        device
                    )

                    # Convert probabilities to logits and compute loss
                    val_probs_tensor = torch.tensor(val_probs, dtype=self.data_type).to(
                        device
                    )

                    # Convert to logits first, then handle extreme values
                    val_logits = torch.logit(val_probs_tensor)

                    # Clip extreme logit values to prevent NaN in loss computation
                    # Using values that are safe for bfloat16
                    val_logits = torch.clamp(val_logits, min=-10.0, max=10.0)

                    # Check for NaN values in logits
                    if torch.isnan(val_logits).any():
                        print("Warning: NaN values detected in validation logits")
                        print("Min logit:", val_logits.min().item())
                        print("Max logit:", val_logits.max().item())
                        val_logits = torch.nan_to_num(val_logits, nan=0.0)

                    val_loss = criterion(val_logits.squeeze(), val_y_tensor).item()

                    # Check for NaN loss
                    if np.isnan(val_loss):
                        print("Warning: NaN validation loss detected")
                        print("Validation probabilities shape:", val_probs_tensor.shape)
                        print("Validation labels shape:", val_y_tensor.shape)
                        val_loss = float(
                            "inf"
                        )  # Set to infinity to avoid selecting this model
                        breakpoint()

                    print(f"Validation loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.model.state_dict().copy()

        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def average_activations(
        self, acts: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        masked_acts = acts * mask[..., None]
        token_counts = mask.sum(dim=1, keepdim=True)
        # Add small epsilon to avoid division by zero
        token_counts = torch.clamp(token_counts, min=1)
        averaged_acts = masked_acts.sum(dim=1) / token_counts
        return averaged_acts

    @torch.no_grad()
    def predict_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size"]:
        """
        Predict the probabilities of the activations.

        Takes the mean before putting through the model.

        Outputs are expected in the shape (batch_size,)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        device = self.training_args["device"]

        # Switch batch norm to eval mode:
        self.model = self.model.to(device)
        self.model.eval()

        acts_tensor = self.average_activations(
            torch.tensor(activations._activations, dtype=self.data_type),
            torch.tensor(activations._attention_mask, dtype=self.data_type),
        )
        logits = self.model(acts_tensor.to(self.data_type).to(device))

        return nn.Sigmoid()(logits).detach().cpu().squeeze().float().numpy()

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

        model = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False),
        )

        # Initialize the linear layer weights to zeros
        torch.nn.init.zeros_(model[1].weight)  # type: ignore

        return model.to(self.data_type)


class AttentionLayer(nn.Module):
    """
    Attention Layer for the attention probe.

    embed_dim: The embedding dimension of the model residual stream

    The module takes the activations from the model residual stream maps it to a single
    dimensional attention embedding dimension to create queries and keys with dims:
    - query: (batch_size, seq_len, 1)
    - key: (batch_size, seq_len, 1)
    - value: (batch_size, seq_len, 1)

    It then passes these through a multi-head attention layer with a single head.

    The output is then squeezed to remove the singleton dimension.

    These design decisions were made to minimize the number of parameters in the
    attention component of the probe.
    """

    def __init__(self, embed_dim: int, attn_hidden_dim: int):
        super().__init__()

        self.query_linear = nn.Linear(embed_dim, attn_hidden_dim)
        self.key_linear = nn.Linear(embed_dim, attn_hidden_dim)
        self.value_linear = nn.Linear(embed_dim, attn_hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_hidden_dim, num_heads=1, batch_first=True
        )

    def forward(
        self,
        activations: Float[torch.Tensor, "batch_size seq_len embed_dim"],
    ) -> Float[torch.Tensor, "batch_size seq_len"]:
        query, key, value = (
            self.query_linear(activations),
            self.key_linear(activations),
            self.value_linear(activations),
        )

        attn_output, _ = self.attn(query, key, value, need_weights=False)

        return attn_output.mean(dim=-1, keepdim=True)


class AttentionProbeAttnWeightLogits(nn.Module):
    def __init__(self, embed_dim: int, attn_hidden_dim: int):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.attention_layer = AttentionLayer(embed_dim, attn_hidden_dim)
        self.linear = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        activations: Float[torch.Tensor, "batch_size seq_len embed_dim"],
    ) -> Float[torch.Tensor, "batch_size seq_len"]:
        """
        The forward pass of the attention probe.

        TODO: swap activation weighting before the attention layer and after the attention layer
        """

        # batch_size, seq_len, _ = activations.shape
        # Flatten activations:
        # activations = einops.rearrange(activations, "b s e -> (b s) e")
        # Based on https://discuss.pytorch.org/t/how-does-the-batch-normalization-work-for-sequence-data/30839/2
        activations = activations.permute(0, 2, 1)
        activations = self.batch_norm(activations)
        activations = activations.permute(0, 2, 1)
        # activations = einops.rearrange(
        #    activations, "(b s) e -> b s e", b=batch_size, s=seq_len
        # )

        attn_output = self.attention_layer(activations)

        # Normalize the attention output using min-max normalization
        # This ensures values are in [0,1] range without the exponential scaling of softmax

        # Normalize the attention output by dividing by the sum:
        attn_output = attn_output / attn_output.sum(dim=1, keepdim=True)

        # Normalize the attention output using a softmax:
        # attn_output = torch.softmax(attn_output / torch.tensor(0.1), dim=1)

        linear_output = self.linear(activations)

        # For debugging:
        # attn_output = torch.ones_like(linear_output)
        return (linear_output * attn_output).squeeze()


class AttentionProbeAttnThenLinear(nn.Module):
    """
    Attention probe that uses a single head attention layer to aggregate the sequence.
    The output is passed through a single linear layer.
    """

    def __init__(self, embed_dim: int, attn_hidden_dim: int):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(embed_dim)

        self.query_linear = nn.Linear(embed_dim, attn_hidden_dim)
        self.key_linear = nn.Linear(embed_dim, attn_hidden_dim)
        self.value_linear = nn.Linear(embed_dim, attn_hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_hidden_dim, num_heads=1, batch_first=True
        )

        self.linear = nn.Linear(attn_hidden_dim, 1, bias=False)

    def forward(
        self,
        activations: Float[torch.Tensor, "batch_size seq_len embed_dim"],
    ) -> Float[torch.Tensor, "batch_size seq_len"]:
        activations = activations.permute(0, 2, 1)
        activations = self.batch_norm(activations)
        activations = activations.permute(0, 2, 1)

        keys, queries, values = (
            self.key_linear(activations),
            self.query_linear(activations),
            self.value_linear(activations),
        )

        attn_output, _ = self.attn(queries, keys, values, need_weights=False)

        linear_output = self.linear(attn_output).squeeze(dim=-1)

        return linear_output


class PytorchAttentionClassifier(PytorchLinearClassifier):
    """
    A linear classifier that uses PyTorch. The sequence is aggregated using a learnt attention mechanism.
    """

    training_args: dict
    model: nn.Module | None = None

    def train(
        self,
        activations: Activation,
        y: Float[np.ndarray, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[np.ndarray, " batch_size"] | None = None,
        print_gradient_norm: bool = False,
    ) -> Self:
        """
        Train the classifier on the activations and labels.

        Training is done on the

        Args:
            activations: The activations to train on.
            y: The labels to train on.
            validation_activations: Optional validation activations.
            validation_y: Optional validation labels.
            print_gradient_norm: Whether to print gradient norm during training.

        Returns:
            Self
        """
        device = self.training_args["device"]

        # Convert labels to tensor
        y_tensor = torch.tensor(y, dtype=self.data_type)

        # Just this bit here is different from the PytorchLinearClassifier
        per_entry_dataset = activations.to_dataset(y=y_tensor, per_token=False)

        # Get the embedding dimension from the averaged activations
        embedding_dim = activations._activations.shape[-1]

        # Create a linear model with the correct embedding dimension
        if self.model is None:
            self.model = self.create_model(
                embedding_dim,
                attn_hidden_dim=self.training_args["attn_hidden_dim"],
                probe_architecture=self.training_args["probe_architecture"],
            ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.training_args["optimizer_args"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_args["epochs"],
            eta_min=self.training_args["optimizer_args"]["lr"]
            * self.training_args["scheduler_decay"],
        )

        criterion = nn.BCEWithLogitsLoss()

        dataloader = DataLoader(
            per_entry_dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=True,
        )

        # Initialize variables for tracking best model
        best_val_loss = float("inf")
        best_model_state = None
        self.best_epoch = None

        # Training loop
        # Enable gradient computation
        with torch.set_grad_enabled(True):
            self.model.train()
            for epoch in range(self.training_args["epochs"]):
                running_loss = 0.0
                pbar = tqdm(
                    dataloader, desc=f"Epoch {epoch + 1}/{self.training_args['epochs']}"
                )
                for batch_idx, batch in enumerate(pbar):
                    acts, mask_batch, _, y = batch

                    # acts = self.average_activations(acts, mask_batch)

                    assert isinstance(y, torch.Tensor), "y must be a torch.Tensor"
                    y_tensor = y.to(device)  # type: ignore

                    # Standard training step for AdamW
                    optimizer.zero_grad()
                    outputs = self.model(
                        acts.to(device).to(self.data_type)
                    )  # batch_size, seq_len

                    # Multiply by the attention mask here:
                    # We multiply the acts by the atteniont mask but the model is currently still learning from padded inputs
                    # So we need to multiply the outputs by the attention mask to remove the padded tokens
                    outputs = outputs * mask_batch.to(device).to(self.data_type)

                    outputs = outputs.mean(
                        dim=-1
                    )  # aggregate the attention weighted logits

                    # Ensure outputs and y_tensor have compatible shapes
                    outputs = outputs.view(-1)  # Flatten to (batch_size,)
                    y_tensor = y_tensor.view(-1)  # Flatten to (batch_size,)

                    loss = criterion(outputs, y_tensor)

                    if print_gradient_norm:
                        print("loss", loss)

                    assert not loss.isnan().any(), "Loss is NaN"

                    loss.backward()

                    # Calculate and print gradient norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    if print_gradient_norm:
                        print(f"gradient norm: {grad_norm.item()}")

                    optimizer.step()
                    loss = loss.item()

                    # Update running loss and progress bar
                    running_loss += loss
                    avg_loss = running_loss / (batch_idx + 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Print epoch summary
                scheduler.step()
                print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

                # Validation step if validation data is provided
                if validation_activations is not None and validation_y is not None:
                    self.model.eval()
                    with torch.no_grad():
                        # Get probabilities for validation data
                        val_probs = self.predict_proba(validation_activations)

                        # Convert validation labels to tensor
                        val_y_tensor = torch.tensor(
                            validation_y, dtype=self.data_type
                        ).to(device)

                        # Convert probabilities to logits and compute loss
                        val_probs_tensor = torch.tensor(
                            val_probs, dtype=self.data_type
                        ).to(device)

                        # Convert to logits first, then handle extreme values
                        val_logits = torch.logit(val_probs_tensor)

                        # Clip extreme logit values to prevent NaN in loss computation
                        # Using values that are safe for bfloat16
                        val_logits = torch.clamp(val_logits, min=-10.0, max=10.0)

                        # Check for NaN values in logits
                        if torch.isnan(val_logits).any():
                            print("Warning: NaN values detected in validation logits")
                            print("Min logit:", val_logits.min().item())
                            print("Max logit:", val_logits.max().item())
                            val_logits = torch.nan_to_num(val_logits, nan=0.0)

                        val_loss = criterion(val_logits.squeeze(), val_y_tensor).item()

                        # Check for NaN loss
                        if np.isnan(val_loss):
                            print("Warning: NaN validation loss detected")
                            print(
                                "Validation probabilities shape:",
                                val_probs_tensor.shape,
                            )
                            print("Validation labels shape:", val_y_tensor.shape)
                            val_loss = float(
                                "inf"
                            )  # Set to infinity to avoid selecting this model

                        print(f"Validation loss: {val_loss:.4f}")

                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model_state = self.model.state_dict().copy()
                            self.best_epoch = epoch + 1  # Store 1-indexed epoch number

        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    @torch.no_grad()
    def predict_token_logits(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size seq_len"]:
        """
        Predict the logits of the activations.

        1. Get the activations
        2. Pass through the model
        3. Multiply by the attention mask
        4. Reshape back to (batch_size, seq_len) and then return.

        Outputs the classifier logits that are expected in the shape (batch_size, seq_len)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        device = self.training_args["device"]

        # Process the activations into a per token dataset to be passed through the model
        batch_size, seq_len, _ = activations.shape

        acts_tensor = activations.get_activations(per_token=False)

        # Switch batch norm to eval mode:
        self.model = self.model.to(device)
        self.model.eval()

        logits = self.model(torch.tensor(acts_tensor, dtype=self.data_type).to(device))

        # Multiply by the attention mask -> to remove padded tokens:
        masked_logits = logits * torch.tensor(
            activations.get_attention_mask(per_token=False),
            dtype=self.data_type,
        ).to(device)

        assert masked_logits.shape == (
            batch_size,
            seq_len,
        ), f"Logits shape is {masked_logits.shape} not {(batch_size, seq_len)}"

        return masked_logits

    def create_model(
        self, embedding_dim: int, attn_hidden_dim: int, probe_architecture: str
    ) -> nn.Module:
        """
        Create a linear model over the embedding dimension dynamically.

        Args:
            embedding_dim: The embedding dimension of the model residual stream
            attn_hidden_dim: The hidden dimension of the attention layer
            probe_architecture: The architecture of the probe

        Probe Architectures:
        "attention_weighted_agg_logits":
            The sequence is aggregated using a weighted sum where the weights are the
            outputs of a learnt attention mechanism.

        "attention_then_linear":
            The sequence is passed through a learnt attention mechanism and then through a
            linear layer.

        Returns:
            A PyTorch module
        """

        # Create an attention probe layer:
        if probe_architecture == "attention_weighted_agg_logits":
            model = AttentionProbeAttnWeightLogits(embedding_dim, attn_hidden_dim).to(
                self.data_type
            )
        elif probe_architecture == "attention_then_linear":
            model = AttentionProbeAttnThenLinear(embedding_dim, attn_hidden_dim).to(
                self.data_type
            )
        else:
            raise NotImplementedError(
                f"Probe architecture {probe_architecture} not implemented"
            )

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, AttentionLayer):
                torch.nn.init.xavier_uniform_(layer.query_linear.weight)
                torch.nn.init.xavier_uniform_(layer.key_linear.weight)
                torch.nn.init.xavier_uniform_(layer.value_linear.weight)

        return model

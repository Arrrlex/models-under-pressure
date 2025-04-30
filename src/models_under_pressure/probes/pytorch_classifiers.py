from dataclasses import dataclass
from typing import Any, Self

import einops
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_under_pressure.config import global_settings
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.probes import aggregations as agg
from models_under_pressure.probes.base import Aggregation


def masked_mean(acts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of the activations, ignoring the elements where the mask is 0.
    """
    batch_size, seq_len, embed_dim = acts.shape
    assert mask.shape == (batch_size, seq_len)
    return acts.sum(dim=1) / mask.sum(dim=1, keepdims=True).clamp(min=1)


@dataclass
class PytorchLinearClassifier:
    """
    A linear classifier that uses PyTorch. The model trains on the flattened batch_size * seq_len
    activations and labels.
    """

    training_args: dict
    aggregation: Aggregation
    model: nn.Module | None = None
    best_epoch: int | None = None
    device: str = global_settings.DEVICE
    dtype: torch.dtype = global_settings.DTYPE

    def setup_for_training(
        self, activations: Activation, **model_kwargs: Any
    ) -> tuple[nn.Module, Activation]:
        """
        Prepare the model and activations for training.
        """
        self.model = self.create_model(activations.embed_dim, **model_kwargs)
        self.model = self.model.to(self.device).to(self.dtype)
        activations = activations.to(self.device, self.dtype)
        return self.model, activations

    def train(
        self,
        activations: Activation,
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
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
        self.model, activations = self.setup_for_training(activations)

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.training_args["optimizer_args"],
        )

        criterion = nn.BCEWithLogitsLoss()

        per_token_dataset = activations.per_token().to_dataset(y)

        dataloader = DataLoader(
            per_token_dataset,
            batch_size=self.training_args["batch_size"],
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
                batch_acts, _, _, batch_y = batch

                optimizer.zero_grad()
                outputs = self.model(batch_acts)
                loss = criterion(outputs.squeeze(), batch_y)
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
                    # Get probabilities for validation data.
                    # Clip extreme logit values to prevent NaN in loss computation
                    # Using values that are safe for bfloat16
                    val_logits = (
                        self.logits(validation_activations).clamp(-10.0, 10.0).view(-1)
                    )

                    val_loss = criterion(
                        val_logits.to(self.device), validation_y
                    ).item()

                    # Check for NaN loss - if it's NaN, set loss to +inf to avoid selecting this model
                    if np.isnan(val_loss):
                        print("Warning: NaN validation loss detected")
                        print(f"{val_logits.shape=}")
                        print(f"{validation_y.shape=}")
                        val_loss = float("inf")

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

    def probs(self, activations: Activation, per_token: bool = False) -> torch.Tensor:
        """
        Predict the probabilities of the activations.

        Outputs are expected in the shape (batch_size,)
        """
        probs = self.logits(activations, per_token=per_token).sigmoid()
        return probs

    @torch.no_grad()
    def logits(
        self, activations: Activation, per_token: bool = False
    ) -> Float[torch.Tensor, " batch_size seq_len"]:
        """
        Predict the logits of the activations.

        If per_token is True, the logits are returned in the shape (batch_size, seq_len),
        with the logits for each token in the sequence.

        If per_token is False, the logits are returned in the shape (batch_size,),
        with the aggregated logit for each sample in the batch.
        """
        if self.model is None:
            raise ValueError("Model not trained")

        batch_size, seq_len, _ = activations.shape

        # Process the activations into a per token dataset
        # Create dummy labels for dataset creation
        dummy_labels = torch.empty(batch_size, device=self.device)
        dataset = activations.per_token().to_dataset(dummy_labels)

        # Create dataloader for batching
        dataloader = DataLoader(
            dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=False,  # No need to shuffle during inference
        )

        # Switch batch norm to eval mode
        self.model.eval()

        # Initialize output tensor
        flattened_logits = torch.zeros(
            (batch_size * seq_len,),
            device=self.device,
            dtype=self.dtype,
        )

        # Create a mask to track which positions in the original sequence are present
        # This will be used to place logits in the correct positions
        attention_mask_flat = activations.attention_mask.view(-1)
        present_indices = torch.where(attention_mask_flat == 1)[0]

        # Process in batches
        start_idx = 0
        for batch_acts, _, _, _ in tqdm(dataloader, desc="Processing batches"):
            mb_size = len(batch_acts)

            # Get logits for this batch
            batch_logits = self.model(batch_acts).squeeze()

            # Get the indices where we should place these logits
            batch_indices = present_indices[start_idx : start_idx + mb_size]

            # Place the logits in the correct positions
            flattened_logits[batch_indices] = batch_logits
            start_idx += mb_size

        # Reshape to (batch_size, seq_len)
        logits = einops.rearrange(
            flattened_logits, "(b s) -> b s", b=batch_size, s=seq_len
        )

        if per_token:
            return logits
        else:
            return self.aggregation(
                logits,
                activations.attention_mask.to(self.device),
                activations.input_ids,
            )

    def create_model(self, embed_dim: int) -> nn.Module:
        """
        Create a linear model over the embedding dimension dynamically.
        """

        # Create a linear model over the embedding dimension
        return nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, 1, bias=False),
        )


@dataclass
class PytorchDifferenceOfMeansClassifier(PytorchLinearClassifier):
    use_lda: bool = False

    def create_model(self, embed_dim: int) -> nn.Module:
        return nn.Linear(embed_dim, 1, bias=False)

    def train(
        self,
        activations: Activation,
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
    ) -> Self:
        self.model, activations = self.setup_for_training(activations)
        mean_acts = masked_mean(activations.activations, activations.attention_mask)

        # Separate positive and negative examples
        pos_acts = mean_acts[y.to(mean_acts.device) == 1].cpu()
        neg_acts = mean_acts[y.to(mean_acts.device) == 0].cpu()

        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        if self.use_lda:
            centered_data = torch.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
            covariance = centered_data.t() @ centered_data / activations.batch_size
            inv = torch.linalg.pinv(covariance.float(), hermitian=True, atol=1e-3).to(
                self.dtype
            )
            param = inv @ direction
        else:
            param = direction

        assert param.shape == (activations.embed_dim,)

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
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
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
        self.model, activations = self.setup_for_training(activations)
        dataset = activations.to_dataset(y)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.training_args["optimizer_args"],
        )

        criterion = nn.BCEWithLogitsLoss()

        dataloader = DataLoader(
            dataset,
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
                batch_acts, batch_mask, _, batch_y = batch

                mean_acts = masked_mean(batch_acts, batch_mask)

                # Standard training step for AdamW
                optimizer.zero_grad()
                outputs = self.model(mean_acts)
                loss = criterion(outputs.squeeze(), batch_y)
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
                    # Clip extreme logit values to prevent NaN in loss computation
                    # Using values that are safe for bfloat16
                    val_logits = self.logits(validation_activations).clamp(-10.0, 10.0)

                    val_loss = criterion(val_logits.squeeze(), validation_y).item()

                    # Check for NaN loss - if it's NaN, set loss to +inf to avoid selecting this model
                    if np.isnan(val_loss):
                        print("Warning: NaN validation loss detected")
                        print(f"{val_logits.shape=}")
                        print(f"{validation_y.shape=}")
                        val_loss = float("inf")

                    print(f"Validation loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.model.state_dict().copy()

        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def probs(self, activations: Activation, per_token: bool = False) -> torch.Tensor:
        return self.logits(activations, per_token=per_token).sigmoid()

    @torch.no_grad()
    def logits(self, activations: Activation, per_token: bool = False) -> torch.Tensor:
        if per_token:
            raise ValueError("Per-token logits not possible for this classifier")

        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()

        mean_acts = masked_mean(activations.activations, activations.attention_mask)

        mean_acts = mean_acts.to(self.device).to(self.dtype)

        logits = self.model(mean_acts)

        return einops.rearrange(logits, "b 1 -> b")

    def create_model(self, embedding_dim: int) -> nn.Module:
        """
        Create a linear model over the embedding dimension dynamically.
        """

        model = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False),
        )

        # Initialize the linear layer weights to zeros
        # todo: check if we actually want this
        torch.nn.init.zeros_(model[1].weight)  # type: ignore

        return model


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


class SimpleAttentionPoolingProbe(nn.Module):
    """
    Correct simple attention-style probe: softmax over dot product with attention vector,
    weighted sum of activations, dot product with output vector, add bias.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_vector = nn.Parameter(torch.randn(embed_dim) * 0.001)
        self.output_vector = nn.Parameter(torch.randn(embed_dim) * 0.001)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        activations: Float[torch.Tensor, "batch_size seq_len embed_dim"],
        mask: Float[torch.Tensor, "batch_size seq_len"] | None = None,
    ) -> Float[torch.Tensor, " batch_size"]:
        device = activations.device
        dtype = activations.dtype
        attn_vector = self.attn_vector.to(device).to(dtype)
        output_vector = self.output_vector.to(device).to(dtype)
        bias = self.bias.to(device).to(dtype)

        batch_size, seq_len, embed_dim = activations.shape

        attn_scores = activations @ attn_vector  # (batch_size, seq_len)

        if mask is not None:
            mask = mask.bool().to(device)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum over the sequence
        weighted_sum = einops.einsum(
            attn_weights,
            activations,
            "batch seq, batch seq hidden_dim -> batch hidden_dim",
        )  # (batch_size, embed_dim)
        logits = (weighted_sum @ output_vector) + bias  # (batch_size,)

        return logits


class PytorchSimpleAttentionClassifier(PytorchLinearClassifier):
    training_args: dict
    model: nn.Module | None = None

    def train(
        self,
        activations: Activation,
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
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
            Self
        """
        self.model, activations = self.setup_for_training(
            activations,
            attn_hidden_dim=self.training_args["attn_hidden_dim"],
            probe_architecture=self.training_args["probe_architecture"],
        )
        dataset = activations.to_dataset(y)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_args["optimizer_args"]["lr"],
        )

        criterion = nn.BCEWithLogitsLoss()

        dataloader = DataLoader(
            dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=True,
        )

        best_model_state = None
        self.best_epoch = None
        best_val_auc = 0

        with torch.set_grad_enabled(True):
            self.model.train()
            for epoch in range(self.training_args["epochs"]):
                running_loss = 0.0
                pbar = tqdm(
                    dataloader, desc=f"Epoch {epoch + 1}/{self.training_args['epochs']}"
                )
                for batch_idx, (batch_acts, batch_mask, _, batch_y) in enumerate(pbar):
                    optimizer.zero_grad()

                    outputs = self.model(batch_acts, batch_mask)
                    outputs = outputs.view(-1)
                    batch_y = batch_y.view(-1)

                    loss = criterion(outputs, batch_y)

                    if print_gradient_norm:
                        print("loss", loss)

                    assert not loss.isnan().any(), "Loss is NaN"

                    loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    if print_gradient_norm:
                        print(f"gradient norm: {grad_norm.item()}")

                    optimizer.step()

                    loss_value = loss.item()
                    running_loss += loss_value
                    avg_loss = running_loss / (batch_idx + 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

                if validation_activations is not None and validation_y is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_logits = self.logits(validation_activations).clamp(
                            -10.0, 10.0
                        )

                        if torch.isnan(val_logits).any():
                            print("Warning: NaN values detected in validation logits")
                            print("Min logit:", val_logits.min().item())
                            print("Max logit:", val_logits.max().item())
                            val_logits = torch.nan_to_num(val_logits, nan=0.0)

                        assert not val_logits.isnan().any(), "Validation logits are NaN"

                        val_logits_flat = val_logits.view(-1).float().cpu().numpy()
                        val_y_flat = validation_y.view(-1).float().cpu().numpy()

                        try:
                            val_auc = roc_auc_score(val_y_flat, val_logits_flat)
                        except ValueError:
                            print("Warning: Unable to compute AUC, setting to 0.0")
                            val_auc = 0.0

                        print(f"Validation AUC: {val_auc:.4f}")

                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                            best_model_state = self.model.state_dict().copy()
                            self.best_epoch = epoch + 1

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self

    def create_model(self, embedding_dim: int) -> nn.Module:
        return SimpleAttentionPoolingProbe(embedding_dim)

    @torch.no_grad()
    def logits(self, activations: Activation, per_token: bool = False) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        batch_size, seq_len, _ = activations.shape

        dummy_labels = torch.empty(batch_size, device=self.device)
        dataset = activations.to_dataset(dummy_labels)

        dataloader = DataLoader(
            dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=False,
        )

        logits = torch.zeros(
            (batch_size),
            device=self.device,
            dtype=self.dtype,
        )

        # Process in batches
        start_idx = 0
        for batch_acts, batch_mask, _, _ in tqdm(dataloader, desc="Processing batches"):
            mb_size = len(batch_acts)

            # Get logits for this batch
            batch_logits = self.model(batch_acts, batch_mask)

            # Store in output tensor
            logits[start_idx : start_idx + mb_size] = batch_logits
            start_idx += mb_size

        return logits
        # Extract the tensor from the Activation object
        # acts_tensor = activations.activations
        # mask = (
        #     activations.attention_mask
        #     if hasattr(activations, "attention_mask")
        #     else None
        # )

        # return self.model(acts_tensor, mask)


class PytorchAttentionClassifier(PytorchLinearClassifier):
    """
    A linear classifier that uses PyTorch. The sequence is aggregated using a learnt attention mechanism.
    """

    training_args: dict
    model: nn.Module | None = None
    aggregation: Aggregation = agg.Mean()

    def train(
        self,
        activations: Activation,
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
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
            Self
        """
        self.model, activations = self.setup_for_training(
            activations,
            attn_hidden_dim=self.training_args["attn_hidden_dim"],
            probe_architecture=self.training_args["probe_architecture"],
        )
        dataset = activations.to_dataset(y)

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
            dataset,
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
                for batch_idx, (batch_acts, batch_mask, _, batch_y) in enumerate(pbar):
                    # Standard training step for AdamW
                    optimizer.zero_grad()
                    outputs = self.model(batch_acts)  # batch_size, seq_len

                    # Multiply by the attention mask here, to avoid learning from padded tokens
                    outputs *= batch_mask

                    # Aggregate the attention weighted logits
                    outputs = outputs.mean(dim=-1)

                    # Ensure outputs and y have compatible shapes
                    outputs = outputs.view(-1)
                    batch_y = batch_y.view(-1)

                    loss = criterion(outputs, batch_y)

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
                        # Clip extreme logit values to prevent NaN in loss computation
                        # Using values that are safe for bfloat16
                        val_logits = self.logits(validation_activations).clamp(
                            -10.0, 10.0
                        )

                        # Check for NaN values in logits
                        if torch.isnan(val_logits).any():
                            print("Warning: NaN values detected in validation logits")
                            print("Min logit:", val_logits.min().item())
                            print("Max logit:", val_logits.max().item())
                            val_logits = torch.nan_to_num(val_logits, nan=0.0)

                        val_loss = criterion(val_logits.squeeze(), validation_y).item()
                        # Check for NaN loss - if found, set loss to +inf to avoid selecting this model
                        if np.isnan(val_loss):
                            print("Warning: NaN validation loss detected")
                            print(f"{val_logits.shape=}")
                            print(f"{validation_y.shape=}")
                            val_loss = float("inf")

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
    def logits(self, activations: Activation, per_token: bool = False) -> torch.Tensor:
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

        self.model.eval()
        batch_size, seq_len, _ = activations.shape

        dummy_labels = torch.empty(batch_size, device=self.device)
        dataset = activations.to_dataset(dummy_labels)

        dataloader = DataLoader(
            dataset,
            batch_size=self.training_args["batch_size"],
            shuffle=False,
        )

        logits = torch.zeros(
            (batch_size, seq_len),
            device=self.device,
            dtype=self.dtype,
        )

        # Process in batches
        start_idx = 0
        for batch_acts, batch_mask, _, _ in tqdm(dataloader, desc="Processing batches"):
            mb_size = len(batch_acts)

            # Get logits for this batch
            batch_logits = self.model(batch_acts) * batch_mask

            # Store in output tensor
            logits[start_idx : start_idx + mb_size] = batch_logits
            start_idx += mb_size

        if per_token:
            return logits
        else:
            return self.aggregation(
                logits,
                activations.attention_mask.to(self.device),
                activations.input_ids,
            )

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
            model = AttentionProbeAttnWeightLogits(embedding_dim, attn_hidden_dim)
        elif probe_architecture == "attention_then_linear":
            model = AttentionProbeAttnThenLinear(embedding_dim, attn_hidden_dim)
        else:
            raise NotImplementedError(
                f"Probe architecture {probe_architecture} not implemented"
            )

        # Set random seed for reproducible initialization
        random_seed = 42
        generator = torch.Generator().manual_seed(random_seed)
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, generator=generator)
            elif isinstance(layer, AttentionLayer):
                torch.nn.init.xavier_uniform_(
                    layer.query_linear.weight, generator=generator
                )
                torch.nn.init.xavier_uniform_(
                    layer.key_linear.weight, generator=generator
                )
                torch.nn.init.xavier_uniform_(
                    layer.value_linear.weight, generator=generator
                )

        return model

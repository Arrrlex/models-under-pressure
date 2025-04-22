from dataclasses import dataclass, field
from typing import Callable, Self

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
from models_under_pressure.probes.aggregations import Mean, Last
from models_under_pressure.config import global_settings


@dataclass
class PytorchLinearClassifier:
    """
    A linear classifier that uses PyTorch. The model trains on the flattened batch_size * seq_len
    activations and labels.
    """

    training_args: dict
    data_type: torch.dtype = field(default=global_settings.DTYPE)
    device: torch.device = field(default=global_settings.DEVICE)
    model: nn.Module | None = None
    best_epoch: int | None = None
    aggregation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = Mean()

    def tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Convert a numpy array or torch tensor to a torch tensor with the correct data type and device.
        """
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=self.data_type, device=self.device)
        else:
            if x.device != self.device:
                x = x.to(self.device)
            if x.dtype != self.data_type:
                x = x.to(self.data_type)
            return x

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

        # Create a linear model
        if self.model is None:
            self.model = self.create_model(activations.shape[2])

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.training_args["optimizer_args"],
        )

        criterion = nn.BCEWithLogitsLoss()

        activations.activations = activations.activations.to(self.device).to(
            self.data_type
        )
        activations.attention_mask = activations.attention_mask.to(self.device).to(
            self.data_type
        )
        activations.input_ids = activations.input_ids.to(self.device).to(self.data_type)

        per_token_dataset = activations.to_dataset(y=y, per_token=True)

        # Calculate class weights to handle imbalanced data
        sample_weights = per_token_dataset.attention_mask

        # Only sample points that are not masked
        sampler = WeightedRandomSampler(
            weights=sample_weights.cpu()
            .numpy()
            .tolist(),  # Convert to list for compatibility
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
                outputs = self.model(acts_tensor)
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
                    val_probs = self.tensor(self.predict_proba(validation_activations))

                    # Convert to logits first, then handle extreme values
                    val_logits = torch.logit(val_probs)

                    # Clip extreme logit values to prevent NaN in loss computation
                    # Using values that are safe for bfloat16
                    val_logits = torch.clamp(val_logits, min=-10.0, max=10.0)

                    # Check for NaN values in logits
                    if torch.isnan(val_logits).any():
                        print("Warning: NaN values detected in validation logits")
                        print("Min logit:", val_logits.min().item())
                        print("Max logit:", val_logits.max().item())
                        val_logits = torch.nan_to_num(val_logits, nan=0.0)

                    val_loss = criterion(val_logits.squeeze(), validation_y).item()

                    # Check for NaN loss
                    if np.isnan(val_loss):
                        print("Warning: NaN validation loss detected")
                        print("Validation probabilities shape:", val_probs.shape)
                        print("Validation labels shape:", validation_y.shape)
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
    ) -> Float[torch.Tensor, " batch_size seq_len"]:
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

        # Process the activations into a per token dataset to be passed through the model
        batch_size, seq_len, _ = activations.shape

        # Flatten the activations and mask

        # Switch batch norm to eval mode:
        self.model.eval()

        # Process in batches
        minibatch_size = 32
        all_logits = []

        for i in range(0, batch_size, minibatch_size):
            batch_acts = activations.activations[i : i + minibatch_size]
            batch_mask = activations.attention_mask[i : i + minibatch_size]

            mb_size = batch_acts.shape[0]  # The last batch may be smaller

            # breakpoint()

            batch_acts = einops.rearrange(batch_acts, "b s e -> (b s) e").to(
                self.device
            )
            batch_mask = einops.rearrange(batch_mask, "b s -> (b s)").to(self.device)

            batch_logits = self.model(batch_acts)
            batch_masked_logits = batch_logits * batch_mask[:, None]
            reshaped_batch_logits = einops.rearrange(
                batch_masked_logits, "(b s) 1 -> b s", b=mb_size, s=seq_len
            )
            all_logits.append(reshaped_batch_logits)

        # Concatenate all batches
        reshaped_logits = torch.cat(all_logits, dim=0)

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
        input_ids = activations.input_ids

        # Take the mean over the sequence length:
        agg_logits = self.aggregation_method(logits, input_ids)

        # Convert the logits to probabilities
        return torch.sigmoid(agg_logits).cpu().float().numpy()

    @torch.no_grad()
    def predict_token_proba(
        self, activations: Activation
    ) -> Float[np.ndarray, " batch_size seq_len"]:
        """
        Predict the probabilities of the activations.

        Outputs are expected in the shape (batch_size, seq_len)
        """
        activations = activations.to(self.device, self.data_type)
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
        model = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False),
        )
        model.to(self.device)
        model.to(self.data_type)

        torch.nn.init.xavier_normal_(model[1].weight)

        return model


@dataclass
class PytorchDifferenceOfMeansClassifier(PytorchLinearClassifier):
    use_lda: bool = False

    def train(
        self,
        activations: Activation,
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
    ) -> Self:
        if validation_activations is not None or validation_y is not None:
            print(
                "Warning: Validation data is not used for PytorchDifferenceOfMeansClassifier"
            )

        acts = activations.activations.to(self.device)
        mask = activations.attention_mask.to(self.device)

        # Apply mask to zero out irrelevant entries
        acts *= mask.unsqueeze(-1)

        # Sum along sequence length and divide by mask sum for each sample
        mask_sums = mask.sum(dim=1, keepdim=True)  # shape: (batch_size, 1)
        averaged_acts = acts.sum(dim=1) / mask_sums  # shape: (batch_size, embed_dim)

        # Separate positive and negative examples
        pos_acts = averaged_acts[y == 1]
        neg_acts = averaged_acts[y == 0]

        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        if self.use_lda:
            centered_data = torch.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
            covariance = centered_data.t() @ centered_data / acts.shape[0]

            inv = torch.linalg.pinv(covariance, hermitian=True, atol=1e-3)
            param = inv @ direction
        else:
            param = direction

        assert param.shape == (activations.embed_dim,)

        param = param.reshape(1, -1)

        self.model = nn.Linear(
            in_features=activations.embed_dim,
            out_features=1,
            bias=False,
            dtype=self.data_type,
            device=self.device,
        )
        self.model.weight.data.copy_(param)

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

        # Just this bit here is different from the PytorchLinearClassifier
        per_entry_dataset = activations.to_dataset(y=y, per_token=False)

        # Get the embedding dimension from the averaged activations
        embed_dim = activations.shape[-1]
        # Create a linear model with the correct embedding dimension

        if self.model is None:
            self.model = self.create_model(embed_dim)

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

                # Standard training step for AdamW
                optimizer.zero_grad()
                outputs = self.model(acts_tensor)
                loss = criterion(outputs.squeeze(), y)
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

                    # Convert probabilities to logits and compute loss
                    val_probs_tensor = self.tensor(val_probs)

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

                    val_loss = criterion(val_logits.squeeze(), validation_y).item()

                    # Check for NaN loss
                    if np.isnan(val_loss):
                        print("Warning: NaN validation loss detected")
                        print("Validation probabilities shape:", val_probs_tensor.shape)
                        print("Validation labels shape:", validation_y.shape)
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
        acts *= mask[..., None]
        token_counts = mask.sum(dim=1, keepdim=True)
        # Add small epsilon to avoid division by zero
        token_counts = torch.clamp(token_counts, min=1)
        averaged_acts = acts.sum(dim=1) / token_counts
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

        # Switch batch norm to eval mode:
        self.model.eval()

        # Process in batches to avoid OOM
        batch_size = 32
        n_samples = len(activations.activations)
        all_logits = []

        for i in range(0, n_samples, batch_size):
            batch_acts = activations.activations[i : i + batch_size].to(self.device)
            batch_mask = activations.attention_mask[i : i + batch_size].to(self.device)

            batch_averaged = self.average_activations(batch_acts, batch_mask)
            batch_logits = self.model(batch_averaged)
            all_logits.append(batch_logits.cpu())

        logits = torch.cat(all_logits, dim=0)

        return nn.Sigmoid()(logits).detach().squeeze().float().numpy()

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
            nn.BatchNorm1d(embedding_dim, dtype=self.data_type, device=self.device),
            nn.Linear(
                embedding_dim, 1, bias=False, dtype=self.data_type, device=self.device
            ),
        )

        # Initialize the linear layer weights to zeros
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


class PytorchAttentionClassifier(PytorchLinearClassifier):
    """
    A linear classifier that uses PyTorch. The sequence is aggregated using a learnt attention mechanism.
    """

    training_args: dict
    model: nn.Module | None = None
    aggregation_method: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = Last()

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

        # Just this bit here is different from the PytorchLinearClassifier
        per_entry_dataset = activations.to_dataset(y=y, per_token=False)

        # Create a linear model with the correct embedding dimension
        if self.model is None:
            self.model = self.create_model(
                activations.embed_dim,
                attn_hidden_dim=self.training_args["attn_hidden_dim"],
                probe_architecture=self.training_args["probe_architecture"],
            )

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

                    # Standard training step for AdamW
                    optimizer.zero_grad()
                    outputs = self.model(acts)

                    # Multiply by the attention mask here:
                    # We multiply the acts by the atteniont mask but the model is currently still learning from padded inputs
                    # So we need to multiply the outputs by the attention mask to remove the padded tokens
                    outputs = outputs * mask_batch

                    outputs = outputs.mean(
                        dim=-1
                    )  # aggregate the attention weighted logits

                    # Ensure outputs and y_tensor have compatible shapes
                    outputs = outputs.view(-1)  # Flatten to (batch_size,)
                    y_tensor = y.view(-1)  # Flatten to (batch_size,)

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
                        val_y_tensor = self.tensor(validation_y)

                        # Convert probabilities to logits and compute loss
                        val_probs_tensor = self.tensor(val_probs)

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
    ) -> Float[torch.Tensor, " batch_size seq_len"]:
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

        # acts = activations.activations.to(self.device)
        # mask = activations.attention_mask.to(self.device)

        # Switch batch norm to eval mode:
        self.model.eval()

        minibatch_size = 32
        logits = torch.zeros(activations.batch_size, activations.seq_len, device="cpu")

        for i in range(0, activations.batch_size, minibatch_size):
            batch_acts = activations.activations[i : i + minibatch_size].to(self.device)
            batch_mask = activations.attention_mask[i : i + minibatch_size].to(
                self.device
            )

            batch_logits = self.model(batch_acts)

            # Multiply by the attention mask -> to remove padded tokens:
            batch_logits *= batch_mask
            logits[i : i + minibatch_size] = batch_logits

        assert (
            logits.shape
            == (
                activations.batch_size,
                activations.seq_len,
            )
        ), f"Logits shape is {logits.shape} not {(activations.batch_size, activations.seq_len)}"

        return logits

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

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, AttentionLayer):
                torch.nn.init.xavier_uniform_(layer.query_linear.weight)
                torch.nn.init.xavier_uniform_(layer.key_linear.weight)
                torch.nn.init.xavier_uniform_(layer.value_linear.weight)

        return model.to(self.device).to(self.data_type)

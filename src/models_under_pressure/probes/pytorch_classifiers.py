from dataclasses import dataclass
from typing import Self

import einops
import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_under_pressure.config import global_settings
from models_under_pressure.interfaces.activations import Activation


@dataclass
class PytorchDifferenceOfMeansClassifier:
    training_args: dict
    model: nn.Module | None = None
    best_epoch: int | None = None
    device: str = global_settings.DEVICE
    dtype: torch.dtype = global_settings.DTYPE
    use_lda: bool = False

    def train(
        self,
        activations: Activation,
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
    ) -> Self:
        self.model = nn.Linear(
            activations.embed_dim, 1, bias=False, device=self.device, dtype=self.dtype
        )

        activations = activations.to(self.device, self.dtype)
        summed_acts = activations.activations.sum(dim=1)
        num_tokens = activations.attention_mask.sum(dim=1, keepdims=True).clamp(min=1)
        mean_acts = summed_acts / num_tokens

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
            return logits.sum(dim=1) / activations.attention_mask.sum(dim=1)


@dataclass(kw_only=True)
class PytorchClassifier:
    training_args: dict
    model: nn.Module | None = None
    best_epoch: int | None = None
    device: str = global_settings.DEVICE
    dtype: torch.dtype = global_settings.DTYPE
    probe_architecture: type[nn.Module]

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
        self.model = self.probe_architecture(
            activations.embed_dim, **self.training_args
        )
        self.model = self.model.to(self.device).to(self.dtype)

        dataset = activations.to_dataset(y)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **self.training_args["optimizer_args"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_args["epochs"],
            eta_min=self.training_args["final_lr"],
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

        # Get gradient accumulation steps from training args, default to 1
        gradient_accumulation_steps = self.training_args.get(
            "gradient_accumulation_steps", 1
        )

        # Training loop
        # Enable gradient computation
        with torch.set_grad_enabled(True):
            self.model.train()
            for epoch in range(self.training_args["epochs"]):
                running_loss = 0.0
                optimizer.zero_grad()  # Zero gradients at the start of each epoch
                pbar = tqdm(
                    dataloader, desc=f"Epoch {epoch + 1}/{self.training_args['epochs']}"
                )
                for batch_idx, (batch_acts, batch_mask, _, batch_y) in enumerate(pbar):
                    # Standard training step for AdamW
                    outputs = self.model(batch_acts, batch_mask)
                    loss = criterion(outputs, batch_y)

                    # Scale loss by gradient accumulation steps
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if print_gradient_norm:
                        print("loss", loss)

                    assert not loss.isnan().any(), "Loss is NaN"

                    # Calculate and print gradient norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    if print_gradient_norm:
                        print(f"gradient norm: {grad_norm.item()}")

                    # Only step optimizer and zero gradients after accumulating enough steps
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    # Update running loss (multiply by gradient_accumulation_steps to get actual loss)
                    running_loss += loss.item() * gradient_accumulation_steps
                    avg_loss = running_loss / (batch_idx + 1)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Print epoch summary
                scheduler.step()
                print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

                # Validation step if validation data is provided
                if validation_activations is not None and validation_y is not None:
                    # Get probabilities for validation data
                    # Clip extreme logit values to prevent NaN in loss computation
                    # Using values that are safe for bfloat16
                    val_logits = self.logits(validation_activations).clamp(-10.0, 10.0)

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

    def probs(self, activations: Activation, per_token: bool = False) -> torch.Tensor:
        return self.logits(activations, per_token).sigmoid()

    @torch.no_grad()
    def logits(self, activations: Activation, per_token: bool = False) -> torch.Tensor:
        """
        Predict the logits of the activations.
        """
        if self.model is None:
            raise ValueError("Model not trained")

        assert not per_token, "Per token is not supported for attention classifier"

        self.model.eval()

        dataloader = DataLoader(
            activations.to_dataset(),
            batch_size=self.training_args["batch_size"],
            shuffle=False,
        )

        logits = []

        # Process in batches
        for batch_acts, batch_mask, _, _ in tqdm(dataloader, desc="Processing batches"):
            logits.append(self.model(batch_acts, batch_mask))

        return torch.cat(logits, dim=0)

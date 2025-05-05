from dataclasses import dataclass
from typing import Protocol, Self

import einops
import torch
import torch.nn as nn
from jaxtyping import Float
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from models_under_pressure.config import global_settings
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.utils import as_numpy


class PytorchClassifier(Protocol):
    training_args: dict
    model: nn.Module | None
    device: str
    dtype: torch.dtype

    def train(
        self,
        activations: Activation,
        y: Float[torch.Tensor, " batch_size"],
        validation_activations: Activation | None = None,
        validation_y: Float[torch.Tensor, " batch_size"] | None = None,
    ) -> Self: ...

    def probs(
        self, activations: Activation, per_token: bool = False
    ) -> torch.Tensor: ...

    def logits(
        self, activations: Activation, per_token: bool = False
    ) -> torch.Tensor: ...


@dataclass
class PytorchDifferenceOfMeansClassifier:
    training_args: dict
    model: nn.Module | None = None
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
        # Use DataLoader to process in batches
        dataset = activations.to_dataset(y)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
        )

        pos_sum = None
        neg_sum = None
        pos_count = 0
        neg_count = 0

        for batch_acts, batch_mask, _, batch_y in tqdm(
            dataloader, desc="Batch mean calc"
        ):
            batch_acts = batch_acts.to(self.device, self.dtype)
            batch_y = batch_y.to(self.device)

            # Masked mean per sample
            summed_acts = (batch_acts * batch_mask.unsqueeze(-1)).sum(dim=1)
            num_tokens = batch_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_acts = summed_acts / num_tokens

            pos_mask = batch_y == 1
            neg_mask = batch_y == 0

            if pos_mask.any():
                pos_acts = mean_acts[pos_mask]
                if pos_sum is None:
                    pos_sum = pos_acts.sum(dim=0)
                else:
                    pos_sum += pos_acts.sum(dim=0)
                pos_count += pos_acts.shape[0]

            if neg_mask.any():
                neg_acts = mean_acts[neg_mask]
                if neg_sum is None:
                    neg_sum = neg_acts.sum(dim=0)
                else:
                    neg_sum += neg_acts.sum(dim=0)
                neg_count += neg_acts.shape[0]

        pos_mean = pos_sum / pos_count
        neg_mean = neg_sum / neg_count
        direction = pos_mean - neg_mean

        assert direction.shape == (activations.embed_dim,)

        self.model = nn.Linear(
            activations.embed_dim, 1, bias=False, device=self.device, dtype=self.dtype
        )

        self.model.weight.data.copy_(direction.reshape(1, -1))

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
class PytorchAdamClassifier:
    training_args: dict
    model: nn.Module | None = None
    best_epoch: int | None = None
    device: str = global_settings.DEVICE
    dtype: torch.dtype = global_settings.DTYPE
    probe_architecture: type[nn.Module]
    wandb_project: str | None = global_settings.WANDB_PROJECT
    wandb_api_key: str | None = global_settings.WANDB_API_KEY

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

        # Initialize wandb if project name is provided
        if self.wandb_project is not None:
            if self.wandb_api_key is not None:
                wandb.login(key=self.wandb_api_key)
            wandb.init(
                project=self.wandb_project,
                config=self.training_args,
            )
            # Log model architecture
            wandb.watch(self.model, log="all")

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
        if validation_y is not None:
            val_y_numpy = as_numpy(validation_y)

        # Initialize variables for tracking best model
        best_val_auroc = 0.0
        best_model_state = None
        self.best_epoch = None
        epochs_without_improvement = 0

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

                    # Log batch metrics to wandb
                    if self.wandb_project is not None:
                        wandb.log(
                            {
                                "batch_loss": loss.item() * gradient_accumulation_steps,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "batch": batch_idx,
                            }
                        )

                # Print epoch summary
                scheduler.step()
                print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

                # Log epoch metrics to wandb
                if self.wandb_project is not None:
                    wandb.log(
                        {
                            "epoch_loss": avg_loss,
                            "epoch": epoch,
                        }
                    )

                # Validation step if validation data is provided
                if validation_activations is not None and validation_y is not None:
                    val_probs = as_numpy(self.probs(validation_activations))
                    auroc = roc_auc_score(val_y_numpy, val_probs)

                    print(f"Validation AUROC: {auroc:.5f}")

                    # Log validation metrics to wandb
                    if self.wandb_project is not None:
                        wandb.log(
                            {
                                "validation_auroc": auroc,
                                "epoch": epoch,
                            }
                        )

                    if auroc > best_val_auroc:
                        best_val_auroc = auroc
                        best_model_state = self.model.state_dict().copy()
                        self.best_epoch = epoch + 1  # Store 1-indexed epoch number
                        epochs_without_improvement = 0

                        # Log best model metrics to wandb
                        if self.wandb_project is not None:
                            wandb.log(
                                {
                                    "best_validation_auroc": auroc,
                                    "best_epoch": epoch + 1,
                                }
                            )
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= self.training_args["patience"]:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            if self.wandb_project is not None:
                                wandb.log(
                                    {
                                        "early_stopping_epoch": epoch + 1,
                                        "final_validation_auroc": auroc,
                                    }
                                )
                            break

        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Close wandb run
        if self.wandb_project is not None:
            wandb.finish()

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

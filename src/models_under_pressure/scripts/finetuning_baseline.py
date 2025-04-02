from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForCausalLM


class ClassifierModule(pl.LightningModule):
    """
    A PyTorch Lightning Module for training a classifier.

    This module provides a flexible framework for training classification models
    with PyTorch Lightning, including training, validation, and testing loops,
    as well as logging metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_params: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize the classifier module.

        Args:
            model: The PyTorch model to train
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            optimizer_class: The optimizer class to use
            scheduler_class: The learning rate scheduler class to use
            scheduler_params: Parameters for the learning rate scheduler
            num_classes: Number of classes for classification (if not specified in model)
            class_weights: Class weights for handling imbalanced datasets
            label_smoothing: Label smoothing factor for the loss function
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_params = scheduler_params or {}
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        # Determine number of classes if not provided
        if self.num_classes is None:
            # Try to infer from the model's output layer
            for module in reversed(list(self.model.modules())):
                if isinstance(module, nn.Linear):
                    self.num_classes = module.out_features
                    break

        # Initialize loss function
        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights, label_smoothing=self.label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # if self.scheduler_class is not None:
        #     scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val_loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }

        return optimizer

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        x, _ = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return {"logits": logits, "probs": probs, "preds": preds}


class LLMModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )

        # Infere the last layer hidden dimension of the loaded model
        # Find the last Linear layer's output dimension by iterating through model layers in reverse
        last_linear_dim = None
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Linear):
                last_linear_dim = module.out_features
                break

        if last_linear_dim is None:
            raise ValueError("Could not find any Linear layers in the model")

        self.num_classes = num_classes
        self.hidden_dim = last_linear_dim
        self.classifier_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, num_classes),
            nn.Softmax(),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        outputs.hidden_states[-1]  # Get the last layer's hidden states

        # Put the last sequence token through a classifier layer:
        return self.classifier_layer(outputs.hidden_states[-1][:, -1, :])


def load_datasets(
    dataset_path: str, cache_dir: Optional[str] = None
) -> None:  # Tuple[DataLoader, DataLoader, DataLoader]:
    """Load the datasets."""
    pass


def train(
    model_name_or_path: str,
    num_classes: int,
    dataset_name: str,
    cache_dir: Optional[str] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler_params: Optional[Dict[str, Any]] = None,
    num_workers: int = 4,
    batch_size: int = 16,
    max_epochs: int = 10,
    shuffle: bool = True,
    wandb_entity: str = "models-under-pressure",
):
    # Create the specific model instance
    model = LLMModel(model_name_or_path, num_classes, cache_dir)

    # Create the classifier module with the model and training parameters
    classifier = ClassifierModule(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_params=scheduler_params,
        num_classes=num_classes,
    )

    # Load the dataset -> sort this out
    train_dataloader, val_dataloader, test_dataloader = load_datasets(
        dataset_name, cache_dir=cache_dir
    )

    # Create a wandb logger
    logger = WandbLogger(project="models-under-pressure", entity=wandb_entity)

    # Setup the pytorch lightning trainer:
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[ModelCheckpoint(monitor="val_loss", mode="min")],
        logger=logger,
    )

    trainer.fit(classifier, train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(classifier, test_dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a classifier model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of classes for classification",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching downloads",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_false",
        dest="shuffle",
        help="Disable shuffling of training data",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="models-under-pressure",
        help="Weights & Biases entity name",
    )

    args = parser.parse_args()

    train(
        model_name_or_path=args.model_name_or_path,
        num_classes=args.num_classes,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        shuffle=args.shuffle,
        wandb_entity=args.wandb_entity,
    )

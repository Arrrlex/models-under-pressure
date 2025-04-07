import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from models_under_pressure.config import (
    EVAL_DATASETS_BALANCED,
    GENERATED_DATASET,
    RESULTS_DIR,
    SYNTHETIC_DATASET_PATH,
    TEST_DATASETS_BALANCED,
)
from models_under_pressure.interfaces.dataset import LabelledDataset


class BaselineResults(BaseModel):
    _logits: Optional[torch.Tensor] = None
    _labels: Optional[torch.Tensor] = None

    """
    An interface for working with the logits and labels collected at test time from the
    model.
    model_name: The name of the model
    dataset_name: The name of the dataset on which the results are collected.
    logits: A tensor of logits across the dataset.
    labels: A tensor of labels across the dataset.
    """

    @property
    def logits(self) -> torch.Tensor | None:
        return self._logits

    @property
    def labels(self) -> torch.Tensor | None:
        return self._labels

    @logits.setter
    def logits(self, value: torch.Tensor):
        assert len(value.shape) == 2, f"Logits must be a 2D tensor, not: {value.shape}"
        assert value.shape[1] == 2, f"Logits 2nd dim must be 2, not: {value.shape[1]}"

        if self._logits is None:
            self._logits = value
        else:
            self._logits = torch.cat((self._logits, value), dim=0)

    @labels.setter
    def labels(self, value: torch.Tensor):
        assert len(value.shape) == 1, f"Labels must be a 1D tensor, not: {value.shape}"

        if self._labels is None:
            self._labels = value
        else:
            self._labels = torch.cat((self._labels, value), dim=0)

    def auroc(self) -> float:
        """Compute the Area Under the Receiver Operating Characteristic Curve."""

        assert (
            self._labels is not None and self._logits is not None
        ), "Labels and logits must be set before computing AUROC"

        sigmoid = torch.nn.Sigmoid()

        return float(
            roc_auc_score(
                self._labels.cpu().numpy(),
                sigmoid(self._logits)[:, 1].cpu().numpy(),
            )
        )

    def tpr_at_fpr(self, fpr: float) -> Tuple[float, float]:
        """Compute the True Positive Rate at a given False Positive Rate."""

        assert (
            self._labels is not None and self._logits is not None
        ), "Labels and logits must be set before computing TPR at FPR"

        sigmoid = torch.nn.Sigmoid()

        fprs, tprs, _ = roc_curve(
            self._labels.cpu().numpy(), sigmoid(self._logits)[:, 1].cpu().numpy()
        )

        # Find the TPR corresponding to the closest FPR
        idx = np.argmin(np.abs(fprs - fpr))
        return float(tprs[idx]), float(fprs[idx])


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
        self.test_results = BaselineResults()

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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""

        assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}, (
            f"batch must contain keys 'input_ids', 'attention_mask', 'labels', "
            f"got {batch.keys()}"
        )

        input_ids, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""

        assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}, (
            f"batch must contain keys 'input_ids', 'attention_mask', 'labels', "
            f"got {batch.keys()}"
        )

        input_ids, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}, (
            f"batch must contain keys 'input_ids', 'attention_mask', 'labels', "
            f"got {batch.keys()}"
        )

        input_ids, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        self.test_results.logits = logits
        self.test_results.labels = y

        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(
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

    def reset_test_results(self):
        self.test_results = BaselineResults()


class LLMModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,  # torch_dtype=torch.bfloat16
        )

        last_linear_dim = None
        # Get hidden size from model config
        last_linear_dim = self.model.config.hidden_size

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


class StakesDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.inputs = df["inputs"].values
        self.labels = (df["labels"] == "high-stakes").astype(int).values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: Union[int, slice]):
        return {"text": self.inputs[idx], "label": self.labels[idx]}


def create_collate_fn(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, max_length: int = 2048
):
    """
    Create a collate function for a pytorch dataloader.
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for a pytorch dataloader.
        """

        # Extract texts and labels from batch
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        # Tokenize the texts
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )  # type: ignore

        # Convert labels to tensor
        labels = torch.tensor(labels)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    return collate_fn


def load_datasets(
    cache_dir: Optional[str] = None,
) -> Tuple[StakesDataset, StakesDataset]:
    """Load the datasets."""

    dataset = LabelledDataset.load_from(
        SYNTHETIC_DATASET_PATH,
        field_mapping=GENERATED_DATASET["field_mapping"],
    )

    df = dataset.to_pandas()

    # Get the high and low stakes situations:
    df["high_stakes_situations"] = df["situations"].apply(lambda x: x["high_stakes"])

    # Splitting the dataset into train and test sets using the high-stake ids is the same as
    # splitting the dataset into train and test sets using the combined situations.
    # Get unique high stakes situations
    unique_high_stakes = df["high_stakes_situations"].unique()

    # Set random seed for reproducibility
    np.random.seed(0)

    # Split unique situations into train and test
    train_test_split = 0.8
    n_train = int(len(unique_high_stakes) * train_test_split)

    # Randomly sample train situations
    train_situations = np.random.choice(unique_high_stakes, size=n_train, replace=False)
    test_situations = np.array(
        [s for s in unique_high_stakes if s not in train_situations]
    )

    # Create train and test masks based on whether high stakes situation is in train set
    train_mask = df["high_stakes_situations"].isin(train_situations)
    test_mask = df["high_stakes_situations"].isin(test_situations)

    # Create train and test datasets
    train_dataset = df[train_mask].reset_index(drop=True)
    test_dataset = df[test_mask].reset_index(drop=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_dataset = StakesDataset(train_dataset)
    test_dataset = StakesDataset(test_dataset)

    return train_dataset, test_dataset


def load_eval_datasets(use_test_set: bool = False) -> List[StakesDataset]:
    """Load the evaluation datasets."""

    if use_test_set:
        # Download the EVAL_DATASETS_BALANCED datasets
        datasets = []
        for dataset_name in TEST_DATASETS_BALANCED:
            datasets.append(
                LabelledDataset.load_from(
                    TEST_DATASETS_BALANCED[dataset_name],
                    field_mapping=GENERATED_DATASET["field_mapping"],
                )
            )
    else:
        datasets = []
        for dataset_name in EVAL_DATASETS_BALANCED:
            datasets.append(
                (
                    LabelledDataset.load_from(
                        EVAL_DATASETS_BALANCED[dataset_name],
                        field_mapping=GENERATED_DATASET["field_mapping"],
                    ),
                    dataset_name,
                )
            )

    # For each dataset, create a StakesDataset:
    torch_datasets = []
    for dataset, dataset_name in datasets:
        dataset = StakesDataset(dataset.to_pandas())
        torch_datasets.append((dataset, dataset_name))
    return torch_datasets


def train(
    model_name_or_path: str,
    num_classes: int,
    cache_dir: Optional[str] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    max_epochs: int = 10,
    shuffle: bool = True,
    wandb_entity: str = "models-under-pressure",
    devices: List[int] = [0],
    use_test_set: bool = False,
    logger: Optional[WandbLogger] = None,
):
    # Create the specific model instance
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
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
    train_dataset, val_dataset = load_datasets(cache_dir=cache_dir)
    collate_fn = create_collate_fn(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

    # Create checkpoint callback:
    best_model_path_template = f"finetune-baselines-{model_name_or_path}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath="/scratch/ucabwjn/models-under-pressure",
        filename=best_model_path_template
        + "-val_loss_{val_loss:.2f}-epoch_{epoch:02d}",
    )

    # Setup the pytorch lightning trainer:
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=devices,
        precision="bf16-true",
        default_root_dir="/scratch/ucabwjn/models-under-pressure",
        callbacks=[checkpoint_callback],  # type: ignore
        logger=logger,
        accumulate_grad_batches=gradient_accumulation_steps,
    )

    trainer.fit(classifier, train_dataloader, val_dataloaders=val_dataloader)

    print(
        f"Loading best model checkpoint from {checkpoint_callback.best_model_path}..."
    )
    classifier = ClassifierModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_params=scheduler_params,
        num_classes=num_classes,
    )
    classifier.eval()

    print("Loading eval datasets...")
    eval_datasets = load_eval_datasets(use_test_set=use_test_set)

    # For each dataset create a DataLoader:
    print("Loading eval dataloaders...")
    dataloaders = []
    for dataset, dataset_name in eval_datasets:
        dataloaders.append(
            (
                DataLoader(
                    dataset,  # type: ignore
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                ),
                dataset_name,
            )
        )

    print("Testing model...")
    output_results = {}
    for i, (test_dataloader, dataset_name) in enumerate(dataloaders):
        print(f"Testing on {dataset_name}, {i} of {len(dataloaders)}")
        trainer.test(classifier, test_dataloader)

        # Save the preds and labels:
        output_results[dataset_name] = classifier.test_results

        # Reset the test results:
        classifier.reset_test_results()

    # For each dataset calculate the AUROC and TPR at FPR=0.1:
    for dataset_name, results in output_results.items():
        auroc = results.auroc()
        tpr_at_fpr = results.tpr_at_fpr(0.1)
        print(f"AUROC for {dataset_name}: {auroc}")
        print(f"TPR at FPR=0.1 for {dataset_name}: {tpr_at_fpr}")

    return output_results


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
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="Devices to use for training",
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
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
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
        default="william_bankes",
        help="Weights & Biases entity name",
    )
    parser.add_argument(
        "--use_test_set",
        action="store_true",
        default=False,
        help="Use the test set for evaluation",
    )

    args = parser.parse_args()

    # Print command line arguments
    print("\nCommand Line Arguments:")
    print("-" * 50)
    for arg, value in vars(args).items():
        print(f"{arg:20} : {value}")
    print("-" * 50 + "\n")

    # Create a wandb logger:
    logger = WandbLogger(
        project="models-under-pressure",
        entity=args.wandb_entity,
        save_dir="/scratch/ucabwjn/models-under-pressure",
    )

    results = train(
        model_name_or_path=args.model_name_or_path,
        num_classes=args.num_classes,
        cache_dir=args.cache_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.max_epochs,
        shuffle=args.shuffle,
        devices=args.devices,
        logger=logger,
        use_test_set=args.use_test_set,
    )

    breakpoint()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        RESULTS_DIR,
        "finetuned_baselines",
        f"{args.model_name_or_path.split('//')[-1]}_{timestamp}.pt",
    )

    torch.save(results, results_path)

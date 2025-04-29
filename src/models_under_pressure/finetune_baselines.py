from datetime import datetime
from typing import Any, Dict, List, Optional, Self, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import grad_norm
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from models_under_pressure.config import DataEfficiencyBaselineConfig, global_settings
from models_under_pressure.interfaces.dataset import BaseDataset, LabelledDataset


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

    @property
    def probits(self) -> torch.Tensor:
        if self._logits is None:
            raise ValueError("Logits must be set before accessing probits")
        sigmoid = torch.nn.Sigmoid()
        return sigmoid(self._logits)[:, 1].cpu().to(torch.float32).numpy()

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

    def accuracy(self) -> float:
        """Compute the accuracy of the model."""
        assert (
            self._labels is not None and self._logits is not None
        ), "Labels and logits must be set before computing accuracy"
        return ((self.probits > 0.5) == self._labels.cpu().numpy()).mean()  # type: ignore

    def auroc(self) -> float:
        """Compute the Area Under the Receiver Operating Characteristic Curve."""

        assert (
            self._labels is not None and self._logits is not None
        ), "Labels and logits must be set before computing AUROC"

        sigmoid = torch.nn.Sigmoid()

        return float(
            roc_auc_score(
                self._labels.cpu().numpy(),
                sigmoid(self._logits)[:, 1].cpu().to(torch.float32).numpy(),
            )
        )

    def tpr_at_fixed_fpr(self, fpr: float) -> Tuple[float, float]:
        """Compute the True Positive Rate at a given False Positive Rate."""

        assert (
            self._labels is not None and self._logits is not None
        ), "Labels and logits must be set before computing TPR at FPR"

        sigmoid = torch.nn.Sigmoid()

        fprs, tprs, _ = roc_curve(
            self._labels.cpu().numpy(),
            sigmoid(self._logits)[:, 1].cpu().to(torch.float32).numpy(),
        )

        # Find the TPR corresponding to the closest FPR
        # Find closest non-zero FPR value
        non_zero_mask = fprs > 0
        try:
            idx = np.argmin(np.abs(fprs[non_zero_mask] - fpr))
            idx = np.where(non_zero_mask)[0][idx]
            return float(tprs[idx]), float(fprs[idx])
        except ValueError:
            print(f"labels: {self._labels.cpu().numpy()}")
            print(
                f"logits: {sigmoid(self._logits)[:, 1].cpu().to(torch.float32).numpy()}"
            )
            print(f"fprs: {fprs}")
            print(f"tprs: {tprs}")
            print("Unable to calculate tpr at fpr")
            return 0.0, 0.0


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

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)


class LLMModel(nn.Module):
    """
    An LLM model with a classifier layer on top of the last hidden state. Used for the finetuning
    baselines.

    Args:
        model_name_or_path: The name or path of the pretrained model to use.
        num_classes: The number of classes for the classification task.
        cache_dir: The directory to cache the model and tokenizer.
    """

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
        # TODO: Is this memory efficient? Implement like Gemma-Shield with specific tokens used as classifier tokens.
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        outputs.hidden_states[-1]  # Get the last layer's hidden states
        # Put the last sequence token through a classifier layer:
        return self.classifier_layer(outputs.hidden_states[-1][:, -1, :])


class StakesDataset(torch.utils.data.Dataset):
    """
    A pytorch dataset wrapper for the labelled dataset class.

    Args:
        df: A pandas dataframe containing the inputs and labels.
    """

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

    Args:
        tokenizer: The tokenizer to use for tokenization.
        max_length: The maximum length of the input texts.
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


class FinetunedClassifier:
    def __init__(self, finetune_config: DataEfficiencyBaselineConfig):
        self.finetune_config = finetune_config
        self._classifier = None
        self._model = None
        self._tokenizer = None
        self._classifier_checkpoint = None

    @property
    def tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        assert (
            self._tokenizer is not None
        ), "Tokenizer must be trained before it can be accessed"
        return self._tokenizer

    @property
    def classifier(self) -> ClassifierModule:
        assert (
            self._classifier is not None
        ), "Classifier must be trained before it can be accessed"
        return self._classifier

    @property
    def model(self) -> LLMModel:
        assert (
            self._model is not None
        ), "Model must be trained before it can be accessed"
        return self._model

    def process_model_configs(self):
        """
        Process the config inputs and raise errors if any are missing.
        """

        model_name_or_path = self.finetune_config.get("model_name_or_path")
        num_classes = self.finetune_config.get("num_classes")
        cache_dir = global_settings.CACHE_DIR

        print(f"Model name or path: {model_name_or_path}")
        print(f"Number of classes: {num_classes}")
        print(f"Saving model to cache dir: {cache_dir}")

        if (model_name_or_path is None) or (num_classes is None):
            raise ValueError(
                "Model name or path and number of classes must be provided in the finetune_config"
            )

        return model_name_or_path, num_classes, cache_dir

    def train(
        self, dataset: LabelledDataset, val_dataset: Optional[LabelledDataset] = None
    ) -> Self:
        """
        Setup the dataset, logger, model checkpointing and train the model using pytorch
        lightning.
        """

        model_name_or_path, num_classes, cache_dir = self.process_model_configs()
        print(f"Cache dir: {cache_dir}")

        # Load the model and tokenizer:
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,  # type: ignore
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model = LLMModel(
            model_name_or_path,
            num_classes,
            cache_dir,
        )

        # Create the pytorch lightning module:
        self._classifier = ClassifierModule(  # TODO: Have I accounted for the attention mask in this code!
            model=self.model,
            num_classes=num_classes,
            **self.finetune_config.get("ClassifierModule", {}),
        )

        # Process the dataset
        collate_fn = create_collate_fn(self.tokenizer)

        # Remove pre-existing activations from the dataset:
        try:  # TODO: Use drop cols as consistent method...
            print("Try removing pre-existing activations from the dataset")
            dataset.remove_field("activations")
            dataset.remove_field("input_ids")
            dataset.remove_field("attention_mask")
            if val_dataset is not None:
                val_dataset.remove_field("activations")
                val_dataset.remove_field("input_ids")
                val_dataset.remove_field("attention_mask")
        except ValueError:
            print("No pre-existing activations to remove")
            pass

        train_dataset = StakesDataset(dataset.to_pandas())
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.finetune_config.get("batch_size", 12),
            shuffle=self.finetune_config.get("shuffle", True),
            collate_fn=collate_fn,
        )

        if val_dataset is not None:
            val_dataset_object = StakesDataset(val_dataset.to_pandas())
            val_dataloader = DataLoader(
                val_dataset_object,
                batch_size=self.finetune_config.get("batch_size", 12),
                shuffle=False,
                collate_fn=collate_fn,
            )

        # Create checkpoint callback:
        best_model_path_template = f"finetune-baselines-{model_name_or_path}"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=cache_dir,
            filename=best_model_path_template + "-{val_loss:.2f}-{epoch:02d}",
        )

        # Create logger:
        if self.finetune_config.get("logger", None) is not None:
            logger = hydra.utils.instantiate(
                self.finetune_config.get("logger"),
                name=f"finetune-baselines-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            )
        else:
            logger = None

        # Setup the pytorch lightning trainer:
        self._trainer = pl.Trainer(
            callbacks=[checkpoint_callback],  # type: ignore
            logger=logger,
            **self.finetune_config.get("Trainer", {}),
        )

        # Train the finetuned model:
        if val_dataset is not None:
            self._trainer.fit(self.classifier, train_dataloader, val_dataloader)
        else:
            self._trainer.fit(self.classifier, train_dataloader)

        # Load the best model checkpoint:
        if val_dataset is not None:
            self._classifier_checkpoint = checkpoint_callback.best_model_path
            print(f"Loading best model checkpoint from: {self._classifier_checkpoint}")
            self._classifier = ClassifierModule.load_from_checkpoint(
                self._classifier_checkpoint,
                model=self.model,
                num_classes=num_classes,
                **self.finetune_config.get("ClassifierModule", {}),
            )
        self._classifier.eval()

        return self

    def predict(self, dataset: BaseDataset) -> list:
        """
        Predict hard labels for each example in the dataset. Uses cutoff of 0.5.
        """

        probits = self.predict_proba(dataset)
        return [1 if p > 0.5 else 0 for p in probits]

    def predict_proba(self, dataset: BaseDataset) -> list:
        """
        Predict the probability of each example in the dataset being high-stakes.
        """

        return self.get_results(dataset).probits.tolist()

    @torch.no_grad()
    def get_results(self, dataset: BaseDataset) -> BaselineResults:
        """
        Using the provided dataset, test the finetuned model and return the BaselineResults
        object.

        Args:
            dataset: The dataset to test the model on.

        Returns:
            The BaselineResults object.
        """
        # Get the classifier, will throw an error if it is not trained:
        classifier = self.classifier
        classifier.reset_test_results()

        # Create a collate function:
        collate_fn = create_collate_fn(self.tokenizer)

        # Remove pre-existing activations from the dataset:
        try:
            print("Try removing pre-existing activations from the dataset")
            dataset.remove_field("activations")
            dataset.remove_field("input_ids")
            dataset.remove_field("attention_mask")
        except ValueError:
            print("No pre-existing activations to remove")
            pass
        # Create a test dataloader:
        test_dataloader = (
            DataLoader(
                StakesDataset(dataset.to_pandas()),
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
            ),
        )

        # Test the model, evaluating on the test set:
        self._trainer.test(self.classifier, test_dataloader)
        return self.classifier.test_results

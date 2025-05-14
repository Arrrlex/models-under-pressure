import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple, Union

import deepspeed
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# from pytorch_lightning.utilities.consolidate_checkpoint import consolidate_checkpoint
# from lightning.fabric.utilities.consolidate_checkpoint import consolidate_checkpoint
from lightning.fabric.utilities.load import _load_distributed_checkpoint

# from pytorch_lightning.fabric.strategies import FSDPStrategy
from pydantic import BaseModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.utilities import grad_norm
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,  # type: ignore
    AutoTokenizer,  # type: ignore
    PreTrainedTokenizer,  # type: ignore
    PreTrainedTokenizerFast,  # type: ignore
)

from models_under_pressure.config import (
    EVAL_DATASETS,
    RESULTS_DIR,
    SYNTHETIC_DATASET_PATH,
    FinetuneBaselineConfig,
    global_settings,
)
from models_under_pressure.dataset_utils import load_dataset, load_train_test
from models_under_pressure.experiments.evaluate_probes import calculate_metrics
from models_under_pressure.interfaces.dataset import (
    BaseDataset,
    LabelledDataset,
    to_dialogue,
)
from models_under_pressure.interfaces.results import FinetunedBaselineResults
from models_under_pressure.utils import hf_login

hf_login()


class BaselineResults(BaseModel):
    _logits: Optional[torch.Tensor] = None
    _labels: Optional[torch.Tensor] = None
    _ids: Optional[List[str]] = None

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
    def ids(self) -> List[str] | None:
        return self._ids

    @property
    def probits(self) -> np.ndarray:
        if self._logits is None:
            raise ValueError("Logits must be set before accessing probits")
        probs = torch.softmax(self._logits, dim=-1)
        return probs[:, 1].cpu().to(torch.float32).numpy()

    @ids.setter
    def ids(self, value: List[str]):
        self._ids = value

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
        batch_size: int = 1,
        class_weights: Optional[torch.Tensor] = None,
        trainer_args: Optional[Dict[str, Any]] = None,
        label_smoothing: float = 0.0,
        optimizer: str = "adam",
    ):
        """
        Initialize the classifier module.

        Args:
            model: The PyTorch model to train
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            scheduler_params: Parameters for the learning rate scheduler
            num_classes: Number of classes for classification (if not specified in model)
            batch_size: Batch size for logging metrics
            class_weights: Class weights for handling imbalanced datasets
            trainer_args: Arguments for the PyTorch Lightning Trainer
            label_smoothing: Label smoothing factor for the loss function
            optimizer: Optimizer to use ("adam" or "adafactor")
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "trainer_args"])

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.scheduler_params = scheduler_params or {}
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.test_results = BaselineResults()
        self.test_outputs = []
        self.trainer_args = trainer_args
        self.optimizer = optimizer

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

        expected_keys = {"input_ids", "attention_mask", "labels"}
        assert expected_keys.issubset(
            batch.keys()
        ), f"batch must contain at least keys {expected_keys}, got {batch.keys()}"

        input_ids, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""

        expected_keys = {"input_ids", "attention_mask", "labels"}
        assert expected_keys.issubset(
            batch.keys()
        ), f"batch must contain at least keys {expected_keys}, got {batch.keys()}"

        input_ids, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return {"val_loss": loss, "val_acc": acc}

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        expected_keys = {"input_ids", "attention_mask", "labels"}
        assert expected_keys.issubset(
            batch.keys()
        ), f"batch must contain at least keys {expected_keys}, got {batch.keys()}"

        input_ids, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, y)

        # Log metrics
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(
            "test_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        # Always store id as a list of strings
        ids = batch["id"]  # already a list of strings
        self.test_outputs.append({"logits": logits, "labels": y, "id": ids})
        return

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Get trainer strategy from trainer
        strategy = self.trainer_args.get("strategy", "") if self.trainer_args else ""

        if strategy.startswith("deepspeed") and strategy.endswith("offload"):
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            if self.optimizer.lower() == "adafactor":
                optimizer = torch.optim.Adafactor(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.optimizer.lower() == "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.optimizer.lower() == "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            elif self.optimizer.lower() == "adamw8bit":
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            else:
                raise ValueError(f"Optimizer {self.optimizer} not supported")

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

    def on_test_epoch_start(self):
        # Clear outputs at the start of each test epoch
        self.test_outputs = []

    def on_test_epoch_end(self):
        # Aggregate outputs for this process
        logits = torch.cat([x["logits"] for x in self.test_outputs])
        labels = torch.cat([x["labels"] for x in self.test_outputs])
        ids = [id_ for x in self.test_outputs for id_ in x["id"]]

        # No all_gather, just store local results
        self.test_results._logits = logits.cpu()
        self.test_results._labels = labels.cpu()
        self.test_results._ids = ids


class LLMModel(nn.Module):
    """
    An LLM model with a classifier layer on top of the last hidden state. Used for the finetuning
    baselines.

    Returns the unnormalized logits for the last token in the sequence.

    Args:
        model_name_or_path: The name or path of the pretrained model to use.
        num_classes: The number of classes for the classification task.
        cache_dir: The directory to cache the model and tokenizer.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        padding_side: str,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,  # Doesn't seem to make a difference
        )

        def _get_hidden_size(model):
            cfg = model.config
            # most models (Llama, GPT-NeoX, Gemma-1B-text, …)
            if hasattr(cfg, "hidden_size"):
                return cfg.hidden_size
            # multimodal wrappers (Gemma-3-4B/12B/27B, Idefics, etc.)
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
                return cfg.text_config.hidden_size
            # always works: read it from the output-embedding matrix
            return model.get_output_embeddings().weight.shape[-1]

        # Get hidden size from model config
        hidden_size = _get_hidden_size(self.model)

        new_head = nn.Linear(hidden_size, num_classes)
        self.model.set_output_embeddings(new_head)

        self.model.config.vocab_size = num_classes
        self.num_classes = num_classes
        self.hidden_dim = hidden_size
        self.padding_side = padding_side

        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.padding_side == "right":
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_idx = torch.arange(seq_lengths.size(0), device=seq_lengths.device)
            outputs = self.model(input_ids, attention_mask=attention_mask).logits
            # print(seq_lengths, outputs.shape)
            return outputs[batch_idx, seq_lengths, :]
        elif self.padding_side == "left":
            return self.model(input_ids, attention_mask=attention_mask).logits[:, -1, :]
        else:
            raise ValueError(f"Padding side {self.padding_side} not supported")


class StakesDataset(torch.utils.data.Dataset):
    """
    A pytorch dataset wrapper for the labelled dataset class.

    Args:
        df: A pandas dataframe containing the inputs and labels.
    """

    def __init__(self, dataset: LabelledDataset):
        df = dataset.to_pandas()
        self.inputs = df["inputs"].values
        self.labels = (df["labels"] == "high-stakes").astype(int).values
        self.ids = [str(i) for i in df["ids"].values]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: Union[int, slice]):
        return {
            "input": self.inputs[idx],
            "label": self.labels[idx],
            "id": self.ids[idx],
        }


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

        # Extract texts, labels, and ids from batch
        dialogues = [item["input"] for item in batch]
        labels = [item["label"] for item in batch]
        ids = [str(item["id"]) for item in batch]

        dialogues = [to_dialogue(d) for d in dialogues]
        input_dicts = [[d.model_dump() for d in dialogue] for dialogue in dialogues]

        input_str = tokenizer.apply_chat_template(
            input_dicts,
            tokenize=False,  # Return string instead of tokens
            add_generation_prompt=False,  # Add final assistant prefix for generation
        )

        # Tokenize the texts
        encoded = tokenizer(
            input_str,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )  # type: ignore

        # Convert labels to tensor
        labels = torch.tensor(labels)
        # Keep ids as list (for string IDs)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "id": ids,  # always a list of strings
        }

    return collate_fn


class FinetunedClassifier:
    best_epoch: int | None = None

    def __init__(self, finetune_config: FinetuneBaselineConfig):
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

    def initialize_model_and_classifier(self):
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
            model_name_or_path=model_name_or_path,
            num_classes=num_classes,
            cache_dir=cache_dir,
            padding_side=self._tokenizer.padding_side,
        )

        # Create the pytorch lightning module:
        self._classifier = ClassifierModule(  # TODO: Have I accounted for the attention mask in this code!
            model=self.model,
            batch_size=self.finetune_config.get("batch_size", 1),
            num_classes=num_classes,
            trainer_args=self.finetune_config.get("Trainer", {}),
            **self.finetune_config.get("ClassifierModule", {}),
        )

    def train(
        self, dataset: LabelledDataset, val_dataset: Optional[LabelledDataset] = None
    ) -> Self:
        """
        Setup the dataset, logger, model checkpointing and train the model using pytorch
        lightning.
        """
        model_name_or_path, num_classes, cache_dir = self.process_model_configs()

        cache_dir = self.initialize_model_and_classifier()
        print(f"Cache dir: {cache_dir}")

        trainer_args = self.finetune_config.get("Trainer", {})

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

        train_dataset = StakesDataset(dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.finetune_config.get("batch_size", 12),
            shuffle=self.finetune_config.get("shuffle", True),
            collate_fn=collate_fn,
            num_workers=self.finetune_config.get("num_workers", 0),
        )

        if val_dataset is not None:
            val_dataset_object = StakesDataset(val_dataset)
            val_dataloader = DataLoader(
                val_dataset_object,
                batch_size=self.finetune_config.get("batch_size", 12),
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.finetune_config.get("num_workers", 0),
            )

        # Create checkpoint callback:
        best_model_path_template = f"finetune-baselines-{model_name_or_path}"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=cache_dir,
            # Removing val_loss part for now as it caused issues due to rounding
            # filename=best_model_path_template + "-{val_loss:.2f}-{epoch:02d}",
            filename=best_model_path_template + "-{epoch:02d}",
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
        if trainer_args.get("strategy", "") == "fsdp":
            trainer_args["strategy"] = FSDPStrategy(
                # fsdp_strategy="FULL_SHARD",
                sharding_strategy="NO_SHARD",
                # mixed_precision="bf16",
                state_dict_type="sharded",
                # state_dict_device="cpu",
            )
        # TODO Take these args from a new field "strategy_args"
        self._trainer = pl.Trainer(
            callbacks=[checkpoint_callback],  # type: ignore
            logger=logger,
            **trainer_args,
        )

        # Train the finetuned model:
        if val_dataset is not None:
            self._trainer.fit(self.classifier, train_dataloader, val_dataloader)
        else:
            self._trainer.fit(self.classifier, train_dataloader)

        batch_size = self.finetune_config.get("batch_size", 1)

        # Load the best model checkpoint:
        if val_dataset is not None:
            self._classifier_checkpoint = checkpoint_callback.best_model_path
            print(f"Loading best model checkpoint from: {self._classifier_checkpoint}")
            strategy = trainer_args.get("strategy", "")

            if isinstance(strategy, FSDPStrategy) or strategy.startswith("fsdp"):
                fresh_llm = LLMModel(
                    model_name_or_path=model_name_or_path,
                    num_classes=num_classes,
                    cache_dir=cache_dir,
                    padding_side=self._tokenizer.padding_side,
                )

                checkpoint_path = self._classifier_checkpoint
                if os.path.isdir(checkpoint_path):
                    full_sd = _load_distributed_checkpoint(Path(checkpoint_path))
                else:  # already a file
                    full_sd = torch.load(checkpoint_path, map_location="cpu")[
                        "state_dict"
                    ]

                self._classifier = ClassifierModule(
                    model=fresh_llm,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    trainer_args=trainer_args,
                    **self.finetune_config.get("ClassifierModule", {}),
                )
                self._classifier.load_state_dict(full_sd, strict=False)
            elif strategy == "deepspeed_stage_2_offload":
                ckpt_dir = checkpoint_callback.best_model_path  # directory
                state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
                self._classifier = ClassifierModule(
                    model=self.model,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    trainer_args=trainer_args,
                    **self.finetune_config.get("ClassifierModule", {}),
                )
                self._classifier.load_state_dict(state_dict, strict=False)
            else:
                self._classifier = ClassifierModule.load_from_checkpoint(
                    self._classifier_checkpoint,
                    model=self.model,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    trainer_args=trainer_args,
                    **self.finetune_config.get("ClassifierModule", {}),
                )
            self.best_epoch = int(
                os.path.splitext(os.path.basename(checkpoint_callback.best_model_path))[
                    0
                ].split("epoch=")[-1]
            )
            self._classifier.to(dtype=torch.bfloat16)
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
        Run a plain forward pass (no PL test‑loop) to avoid the
        dead‑lock that occurs when re‑using a Trainer after ``fit()``.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device).eval()  # inference mode

        # self.classifier.eval()  # ← inference mode
        self.classifier.reset_test_results()

        device = next(self.classifier.parameters()).device
        collate_fn = create_collate_fn(self.tokenizer)

        # Remove pre-existing activations from the dataset:
        try:  # TODO: Use drop cols as consistent method..
            print("Try removing pre-existing activations from the dataset")
            dataset.remove_field("activations")
            dataset.remove_field("input_ids")
            dataset.remove_field("attention_mask")
        except ValueError:
            print("No pre-existing activations to remove")
            pass

        # A single‑worker DataLoader → no inter‑process barriers
        loader = DataLoader(
            StakesDataset(dataset),
            batch_size=self.finetune_config.get("test_batch_size", 1),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = self.classifier(input_ids, attention_mask)

            # accumulate
            self.classifier.test_results.logits = logits.cpu()
            self.classifier.test_results.labels = labels.cpu()
            if self.classifier.test_results._ids is None:
                self.classifier.test_results._ids = batch["id"]
            else:
                self.classifier.test_results._ids.extend(batch["id"])

        return self.classifier.test_results

    def get_full_results(
        self,
        eval_dataset: LabelledDataset,
        dataset_name: str,
        eval_dataset_path: Path,
        max_samples: Optional[int] = None,
    ) -> FinetunedBaselineResults:
        print(f"Getting full results for {dataset_name} ...")

        # Make sure that the dataset doesn't contain duplicate IDs, as this causes issues
        assert len(eval_dataset.ids) == len(
            set(eval_dataset.ids)
        ), "Dataset contains duplicate IDs!"

        # Get the results (only rank 0 will have them)
        results = self.get_results(eval_dataset)
        if results.logits is None or results.labels is None or results.ids is None:
            # Not rank 0, skip aggregation
            import warnings

            warnings.warn(
                "get_full_results called on non-rank-0 process; returning None."
            )
            return None

        # Convert tensors to lists for BaselineResults
        result_ids = list(results.ids) if results.ids is not None else []
        labels = (
            results.labels.cpu().numpy().tolist() if results.labels is not None else []
        )
        scores = results.probits.tolist() if results.logits is not None else []

        # Sort results according to the order of eval_dataset.ids
        eval_ids = list(eval_dataset.ids)
        if set(result_ids) == set(eval_ids):
            # Map result_ids to their indices
            id_to_index = {id_: i for i, id_ in enumerate(result_ids)}
            sort_indices = [id_to_index[id_] for id_ in eval_ids]
            labels = [labels[i] for i in sort_indices]
            scores = [scores[i] for i in sort_indices]
            sorted_ids = [result_ids[i] for i in sort_indices]

            assert (
                sorted_ids == eval_ids
            ), f"Sorted ids: {sorted_ids}, eval_ids: {eval_ids}"
        else:
            raise ValueError(
                f"Mismatch between result IDs and eval_dataset IDs.\nResult IDs: {result_ids}\nEval IDs: {eval_ids}"
            )

        ground_truth = eval_dataset.labels_numpy().tolist()
        assert (
            labels == ground_truth
        ), f"Labels and ground truth are not aligned, so something is wrong here! (labels: {labels}, ground_truth: {ground_truth})"

        # Get the token counts using StakesDataset
        stakes_dataset = StakesDataset(eval_dataset)
        collate_fn = create_collate_fn(self.tokenizer)
        token_counts = [
            len(
                collate_fn(
                    [
                        {
                            "input": sample["input"],
                            "label": sample["label"],
                            "id": sample["id"],
                        }
                    ]
                )["input_ids"][0]
            )
            for sample in stakes_dataset
        ]

        # Create BaselineResults instance
        predicted_labels = [score > 0.5 for score in scores]
        metrics = calculate_metrics(
            y_true=ground_truth,
            y_pred=np.array(scores),
            fpr=0.01,
        )
        baseline_results = FinetunedBaselineResults(
            ids=eval_dataset.ids,
            labels=predicted_labels,
            accuracy=sum(
                [label == gt for label, gt in zip(predicted_labels, ground_truth)]
            )
            / len(predicted_labels),
            metrics=metrics,
            scores=scores,
            ground_truth=ground_truth,
            ground_truth_scale_labels=list(eval_dataset.other_fields["scale_labels"])  # type: ignore
            if "scale_labels" in eval_dataset.other_fields
            else None,
            dataset_name=dataset_name,
            dataset_path=eval_dataset_path,
            model_name=self.finetune_config.model_name_or_path,
            max_samples=max_samples,
            token_counts=token_counts,
            best_epoch=self.best_epoch,
            finetune_config=self.finetune_config,
        )
        return baseline_results


def run_sanity_check(
    finetune_config: FinetuneBaselineConfig,
    dataset_path: Path,
    checkpoint_path: Path | None = None,
    max_samples: Optional[int] = None,
    compute_activations: bool = True,
) -> None:
    finetune_baseline = FinetunedClassifier(finetune_config)

    dataset = load_dataset(
        dataset_path=dataset_path,
        model_name=None,
        layer=None,
        compute_activations=compute_activations,
        n_per_class=max_samples // 2 if max_samples else None,
    )

    # Train the finetune baseline:
    print("Training finetuned baseline...")
    finetune_baseline.train(
        dataset,
        val_dataset=dataset,
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return []

    eval_result = finetune_baseline.get_full_results(
        eval_dataset=dataset,
        dataset_name="sanity_check",
        eval_dataset_path=dataset_path,
        max_samples=max_samples,
    )
    print(eval_result.metrics)


def get_finetuned_baseline_results(
    finetune_config: FinetuneBaselineConfig,
    eval_datasets: dict[str, Path],
    results_dir: Path | None = None,
    train_dataset_path: Path | None = None,
    checkpoint_path: Path | None = None,
    max_samples: Optional[int] = None,
    compute_activations: bool = True,
    use_validation_set: bool = True,
) -> List[FinetunedBaselineResults]:
    assert (
        train_dataset_path is not None or checkpoint_path is not None
    ), "Must provide either train_dataset_path or checkpoint_path"

    finetune_baseline = FinetunedClassifier(finetune_config)

    if checkpoint_path is not None:
        finetune_baseline.initialize_model_and_classifier()
        # Now you can safely load the checkpoint
        print("Loading checkpoint")
        trainer_strategy = finetune_config.get("Trainer", {}).get("strategy", "")

        if trainer_strategy.startswith("fsdp"):
            if os.path.isdir(checkpoint_path):
                full_sd = _load_distributed_checkpoint(Path(checkpoint_path))
                finetune_baseline.classifier.load_state_dict(full_sd, strict=False)
            else:  # already a file
                full_sd = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
                finetune_baseline.classifier.load_state_dict(full_sd, strict=False)
        elif trainer_strategy == "deepspeed_stage_2_offload":
            state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
            finetune_baseline.classifier.load_state_dict(state_dict, strict=False)
        else:
            finetune_baseline.classifier.load_state_dict(torch.load(checkpoint_path))

        # Convert model weights to bfloat16 to match initialization
        finetune_baseline.classifier.to(dtype=torch.bfloat16)

        # Set model to evaluation mode
        finetune_baseline.classifier.eval()

    if train_dataset_path is not None:
        print("Loading train dataset")
        train_dataset, val_dataset = load_train_test(
            dataset_path=train_dataset_path,
            model_name=None,
            layer=None,
            compute_activations=compute_activations,
            n_per_class=max_samples // 2 if max_samples else None,
        )
        print("Training finetuned baseline...")

        # Train the finetune baseline:
        finetune_baseline.train(
            train_dataset,
            val_dataset=val_dataset if use_validation_set else None,
        )

        # torch.cuda.empty_cache()
        del train_dataset
        del val_dataset

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return []

    print("\nLoading eval datasets")
    # We'll use the first eval dataset for the BaselineResults
    eval_results = []
    for dataset_name, eval_dataset_path in tqdm(eval_datasets.items()):
        eval_dataset = load_dataset(
            dataset_path=eval_dataset_path,
            model_name=None,
            layer=None,
            compute_activations=compute_activations,
            n_per_class=max_samples // 2 if max_samples else None,
        )
        eval_result = finetune_baseline.get_full_results(
            eval_dataset=eval_dataset,
            dataset_name=dataset_name,
            eval_dataset_path=eval_dataset_path,
            max_samples=max_samples,
        )
        eval_results.append(eval_result)

        # After each eval, clear the cache, delete the baseline model and dataset subset.
        # torch.cuda.empty_cache()
        del eval_dataset

        if results_dir is not None:
            print(
                f"Saving results for {dataset_name} to {results_dir / 'finetuning.jsonl'}"
            )
            with open(results_dir / "finetuning.jsonl", "a") as f:
                f.write(eval_result.model_dump_json() + "\n")

    return eval_results


def check_collate_fn():
    import argparse
    import random

    from models_under_pressure.config import EVAL_DATASETS
    from models_under_pressure.dataset_utils import load_dataset

    default_dataset_path = EVAL_DATASETS["toolace"]
    default_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"

    parser = argparse.ArgumentParser(
        description="Test collate function and print a random sample from a dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_dataset_path,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=default_model_name_or_path,
        help="Model name or path for tokenizer.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for tokenization.",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(
        dataset_path=args.dataset_path,
        model_name=None,
        layer=None,
        compute_activations=True,  # Avoid loading activations
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create collate function
    collate_fn = create_collate_fn(tokenizer, max_length=args.max_length)

    # Wrap dataset in StakesDataset
    stakes_dataset = StakesDataset(dataset)

    # Get a random sample
    idxs = list(range(len(stakes_dataset)))
    random.shuffle(idxs)
    sample_batch = [stakes_dataset[idxs[0]]]
    collated = collate_fn(sample_batch)

    print("Random sample from collated batch:")
    for k, v in collated.items():
        print(f"{k}: {v}")

    # Decode input_ids back to text (including special tokens)
    decoded_text = tokenizer.decode(collated["input_ids"][0], skip_special_tokens=False)
    print("\nDecoded input_ids (including special tokens):")
    print(decoded_text)


def run_finetune_baselines():
    seeds = [0, 1, 2]
    finetune_model = "Llama-3.1-8B-Instruct"

    for seed in seeds:
        pl.seed_everything(seed)

        # Should be defined via a hydra run config file:
        finetune_config = FinetuneBaselineConfig(
            model_name_or_path="meta-llama/" + finetune_model,
            num_classes=2,
            ClassifierModule={  # set here to the default values
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "scheduler_params": None,
                "class_weights": None,
                "label_smoothing": 0.0,
                "optimizer": "adafactor",  # Can be "adam" or "adafactor"
            },
            batch_size=1,
            shuffle=True,
            logger={
                "_target_": "pytorch_lightning.loggers.WandbLogger",
                "project": "models-under-pressure",
            },
            Trainer={
                "max_epochs": 40,  # 20,
                "accelerator": "gpu",
                "devices": [0],
                "precision": "bf16-true",
                "default_root_dir": global_settings.PL_DEFAULT_ROOT_DIR,
                "accumulate_grad_batches": 16,
            },
        )

        results = get_finetuned_baseline_results(
            finetune_config,
            SYNTHETIC_DATASET_PATH,
            EVAL_DATASETS,
            compute_activations=False,
            max_samples=4000,
        )

        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_path = (
            RESULTS_DIR
            / f"finetune_baselines_{finetune_model}__{seed}_{timestamp}.jsonl"
        )
        for result in results:
            result.save_to(output_path)


if __name__ == "__main__":
    run_finetune_baselines()

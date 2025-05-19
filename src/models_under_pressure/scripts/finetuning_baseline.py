import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)

from models_under_pressure.baselines.finetune import (
    ClassifierModule,
    LLMModel,
    StakesDataset,
    create_collate_fn,
)
from models_under_pressure.config import (
    EVAL_DATASETS_BALANCED,
    RESULTS_DIR,
    SYNTHETIC_DATASET_PATH,
    TEST_DATASETS_BALANCED,
)
from models_under_pressure.interfaces.dataset import LabelledDataset


def load_datasets(
    cache_dir: Optional[str] = None,
) -> Tuple[StakesDataset, StakesDataset]:
    """Load the datasets."""

    dataset = LabelledDataset.load_from(SYNTHETIC_DATASET_PATH)

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
                LabelledDataset.load_from(TEST_DATASETS_BALANCED[dataset_name])
            )
    else:
        datasets = []
        for dataset_name in EVAL_DATASETS_BALANCED:
            datasets.append(
                (
                    LabelledDataset.load_from(EVAL_DATASETS_BALANCED[dataset_name]),
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
    wandb_entity: str = "xxx",
    devices: List[int] = [0],
    use_test_set: bool = False,
    logger: Optional[WandbLogger] = None,
):
    """
    Train a classifier model using the pytorch lightning framework. The dataset loads the
    synthetic dataset, splits it into a train and val set, then loads the eval dev split datasets.
    The script returns a results object which contains the AUROC and TPR at FPR-0.1 for each eval dataset.

    Args:
        model_name_or_path: The name or path of the pretrained model to use.
        num_classes: The number of classes for the classification task.
        cache_dir: The directory to cache the model and tokenizer.
        learning_rate: The learning rate for the optimizer.
        weight_decay: The weight decay for the optimizer.
        scheduler_params: The parameters for the scheduler.
        batch_size: The batch size for the training loop.
        gradient_accumulation_steps: The number of gradient accumulation steps.
        max_epochs: The maximum number of training epochs.
        shuffle: Whether to shuffle the training data.
        wandb_entity: The entity name for wandb.
        devices: The devices to use for training.
        use_test_set: Whether to use the test set for evaluation.
        logger: The logger to use for logging.
    """

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
        dirpath="/scratch/xxx/models-under-pressure",
        filename=best_model_path_template
        + "-val_loss_{val_loss:.2f}-epoch_{epoch:02d}",
    )

    # Setup the pytorch lightning trainer:
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=devices,
        precision="bf16-true",
        default_root_dir="/scratch/xxx/models-under-pressure",
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
        tpr_at_fpr = results.tpr_at_fixed_fpr(0.1)
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
        save_dir="/scratch/xxx/models-under-pressure",
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

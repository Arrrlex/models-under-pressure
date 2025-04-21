import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import optuna
import torch
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import (
    DATA_DIR,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.interfaces.activations import (
    Activation,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchAttentionClassifier,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe


def save_activation(activation: Activation, path: Path) -> None:
    """Save activation object's components as numpy arrays in half precision"""
    np.savez_compressed(
        path,
        activations=activation.activations.numpy().astype(np.float16),
        attention_mask=activation.attention_mask.numpy().astype(np.float16),
        input_ids=activation.input_ids.numpy().astype(np.int32),
    )


def load_activation(path: Path) -> Activation:
    """Load activation object from numpy arrays"""
    data = np.load(path)
    return Activation(
        activations=torch.from_numpy(data["activations"]).float(),
        attention_mask=torch.from_numpy(data["attention_mask"]).float(),
        input_ids=torch.from_numpy(data["input_ids"]).int(),
    )


def compute_and_cache_activations(
    model: LLMModel, datasets: Dict[str, LabelledDataset], layer: int, cache_dir: Path
) -> Dict[str, tuple[Activation, np.ndarray]]:
    """Compute and cache activations for each dataset"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    activations_dict = {}

    for name, dataset in datasets.items():
        cache_path = cache_dir / f"{name}_layer{layer}_activations.npz"
        labels_path = cache_dir / f"{name}_labels.npy"

        if cache_path.exists() and labels_path.exists():
            print(f"Loading cached activations for {name}...")
            activations = load_activation(cache_path)
            labels = np.load(labels_path)
        else:
            print(f"Computing activations for {name}...")
            activations = model.get_batched_activations(dataset=dataset, layer=layer)
            labels = dataset.labels_numpy()

            # Cache the results
            save_activation(activations, cache_path)
            np.save(labels_path, labels)

        activations_dict[name] = (activations, labels)

    return activations_dict


def load_train_val_datasets(
    train_dataset: LabelledDataset,
) -> Tuple[LabelledDataset, LabelledDataset]:
    """Load the training dataset and split it into a train and validation set."""

    df = train_dataset.to_pandas()

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
    train_split_dataset = df[train_mask].reset_index(drop=True)
    test_split_dataset = df[test_mask].reset_index(drop=True)

    print(f"Train dataset size: {len(train_split_dataset)}")
    print(f"Test dataset size: {len(test_split_dataset)}")

    train_dataset = LabelledDataset.from_pandas(train_split_dataset)
    test_dataset = LabelledDataset.from_pandas(test_split_dataset)

    return train_dataset, test_dataset


def tune_probe_hyperparams(
    train_dataset_path: Path,
    layer: int = 11,
    max_samples: int | None = None,
    cache_dir: Path = DATA_DIR / "temp/probe_comparison",
    hyperparams: Dict[str, Any] | None = None,
    probe_architecture: str = "attention_weighted_agg_logits",
):
    # Set random seeds for reproducibility
    np.random.seed(42)

    # Check if all cached files exist first
    cache_dir.mkdir(parents=True, exist_ok=True)

    # List the dataset names:
    all_dataset_names = ["train", "val"]
    all_cached = True

    # Create the cache paths:
    for name in all_dataset_names:
        cache_path = cache_dir / f"{name}_layer{layer}_activations.npz"
        labels_path = cache_dir / f"{name}_labels.npy"
        if not (cache_path.exists() and labels_path.exists()):
            all_cached = False
            break

    # Load cached activations if they all exist
    if all_cached:
        print("All activations found in cache, loading...")
        activations_dict = {
            name: (
                load_activation(cache_dir / f"{name}_layer{layer}_activations.npz"),
                np.load(cache_dir / f"{name}_labels.npy"),
            )
            for name in all_dataset_names
        }
        model = None
    else:
        # Original model and dataset loading logic
        print("Some activations missing, loading model and datasets...")
        model = LLMModel.load(model_name=LOCAL_MODELS["llama-1b"])

        train_dataset = load_filtered_train_dataset(
            dataset_path=train_dataset_path, max_samples=max_samples
        )
        # Split the train dataset into a train and validation set:
        train_dataset, val_dataset = load_train_val_datasets(train_dataset)

        datasets = {"train": train_dataset, "val": val_dataset}
        activations_dict = compute_and_cache_activations(
            model=model, datasets=datasets, layer=layer, cache_dir=cache_dir
        )

    # Use provided hyperparameters or default values
    if hyperparams is None:
        hyperparams = {
            "batch_size": 128,
            "epochs": 40,
            "learning_rate": 1e-2,
            "weight_decay": 0.001,
            "attn_hidden_dim": 1,
            "scheduler_decay": 0.1,
        }

    adamw_args = {
        "batch_size": hyperparams["batch_size"],
        "epochs": hyperparams["epochs"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "attn_hidden_dim": hyperparams["attn_hidden_dim"],
        "scheduler_decay": hyperparams["scheduler_decay"],
        "optimizer_args": {
            "lr": hyperparams["learning_rate"],
            "weight_decay": hyperparams["weight_decay"],
        },
        "probe_architecture": probe_architecture,
    }

    pytorch_probe = PytorchProbe(
        _llm=model,  # type: ignore
        layer=layer,
        hyper_params={},  # Doesn't apply when giving the classifier
        _classifier=PytorchAttentionClassifier(
            training_args=adamw_args,
        ),
    )

    try:
        # Train both probes on training activations
        train_activations, train_labels = activations_dict["train"]
        pytorch_probe._fit(train_activations, train_labels)

        # Evaluate on eval datasets
        eval_activations, eval_labels = activations_dict["val"]

        # Get predictions from both probes
        pytorch_probs = pytorch_probe._classifier.predict_proba(eval_activations)  # type: ignore

        # Calculate metrics
        pytorch_auroc = roc_auc_score(eval_labels, pytorch_probs)

        print(f"Pytorch AUROC: {pytorch_auroc:.4f}")

    except AssertionError as e:
        print(f"Error: {e}")
        pytorch_auroc = 0.0

    return pytorch_auroc


def create_objective(probe_architecture: str) -> Callable[[optuna.Trial], float]:
    """Create an objective function for Bayesian optimization.

    Args:
        probe_architecture: The architecture of the probe to use

    Returns:
        A function that takes a trial object and returns the AUROC score
    """

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Bayesian optimization.

        Args:
            trial: Optuna trial object that suggests hyperparameters

        Returns:
            float: The AUROC score to maximize
        """
        # Define the hyperparameter search space
        hyperparams = {
            "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
            "epochs": trial.suggest_int("epochs", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "scheduler_decay": trial.suggest_float("scheduler_decay", 0.1, 0.99),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "attn_hidden_dim": trial.suggest_int("attn_hidden_dim", 1, 32, step=1),
        }

        # Create a temporary directory for caching
        cache_dir = Path("temp/bayesian_opt_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # try:
        # Run the probe training with suggested hyperparameters
        auroc = tune_probe_hyperparams(
            train_dataset_path=SYNTHETIC_DATASET_PATH,
            max_samples=None,
            cache_dir=cache_dir,
            hyperparams=hyperparams,
            probe_architecture=probe_architecture,
        )
        return float(auroc)

    return objective


def optimize_hyperparameters(
    n_trials: int = 100, probe_architecture: str = "attention_weighted_agg_logits"
) -> Dict[str, Any]:
    """Run Bayesian optimization to find the best hyperparameters.

    Args:
        n_trials: Number of optimization trials to run

    Returns:
        Dict containing the best hyperparameters found
    """
    # Create a study object and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(create_objective(probe_architecture), n_trials=n_trials)

    # Print the best results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for probes"
    )
    parser.add_argument(
        "--probe_arch",
        type=str,
        required=False,
        choices=["attention_weighted_agg_logits", "attention_then_linear"],
        default="attention_weighted_agg_logits",
        help="Architecture of the probe to use",
    )
    args = parser.parse_args()

    # Run the optimization
    best_params = optimize_hyperparameters(
        n_trials=100, probe_architecture=args.probe_arch
    )

    # tune_probe_hyperparams(
    #     train_dataset_path=SYNTHETIC_DATASET_PATH,
    #     max_samples=None,
    #     hyperparams={
    #         "batch_size": 32,
    #         "epochs": 58,
    #         "device": "cuda",
    #         "attn_hidden_dim": 14,
    #         "scheduler_decay": 0.1312581742002716,
    #         "learning_rate": 0.01,
    #         "weight_decay": 0.008003599217166068,
    #     },
    # )

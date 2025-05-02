import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
from pydantic import BaseModel
from tqdm import tqdm

from models_under_pressure.baselines.finetune import (
    FinetunedClassifier,
)
from models_under_pressure.config import (
    EVAL_DATASETS_BALANCED,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
    FinetuneBaselineConfig,
    global_settings,
)
from models_under_pressure.dataset_utils import (
    load_dataset,
    load_train_test,
)
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
)


class FinetuneHyperConfig(BaseModel):
    model_name: str
    layer: int
    dataset_path: Path
    compute_activations: bool = False
    eval_dataset_paths: list[Path]
    model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    num_classes: int = 2
    ClassifierModule: dict = {}
    batch_size: int = 4
    shuffle: bool = True
    logger: Optional[dict] = None
    Trainer: dict = {}


class FinetuneHyperOptResults(BaseModel):
    params: Dict[str, Any]
    auroc: float
    accuracy: float
    tpr_at_fpr: float
    auroc_std: float
    accuracy_std: float
    tpr_at_fpr_std: float
    runs: List[Dict[str, float]]


def load_hyperopt_datasets(
    config: FinetuneHyperConfig,
) -> tuple[LabelledDataset, LabelledDataset, list[LabelledDataset]]:
    """Load the train, val and eval datasets for the hyperopt experiment.

    Args:
        config: The configuration for the hyperopt experiment.

    Returns:
        A tuple of the train, val and eval datasets.
    """
    train_dataset, val_dataset = load_train_test(
        dataset_path=config.dataset_path,
        # model_name=config.model_name,
        # layer=config.layer,
        compute_activations=config.compute_activations,
    )

    eval_datasets = []
    for eval_dataset_path in config.eval_dataset_paths:
        eval_datasets.append(
            load_dataset(
                dataset_path=eval_dataset_path,
                # model_name=config.model_name,
                # layer=config.layer,
                compute_activations=config.compute_activations,
            )
        )

    return train_dataset, val_dataset, eval_datasets


def get_finetune_config(config: FinetuneHyperConfig) -> FinetuneBaselineConfig:
    """Convert FinetuneHyperConfig to FinetuneBaselineConfig.

    Args:
        config: The hyperopt configuration

    Returns:
        FinetuneBaselineConfig for use with the baseline model
    """
    return FinetuneBaselineConfig(
        model_name_or_path=config.model_name_or_path,
        num_classes=config.num_classes,
        ClassifierModule=config.ClassifierModule,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        logger=config.logger,
        Trainer=config.Trainer,
    )


def run_finetune_baseline(
    train_dataset: LabelledDataset,
    val_dataset: LabelledDataset,
    eval_datasets: list[LabelledDataset],
    config: FinetuneHyperConfig,
    num_runs: int = 3,
) -> FinetuneHyperOptResults:
    """Run data efficiency experiment by training probes on different sized subsets of the dataset.

    Args:
        train_dataset: The training dataset
        val_dataset: The validation dataset
        eval_datasets: List of evaluation datasets
        config: Configuration for the data efficiency experiment
        num_runs: Number of runs to perform with the same configuration

    Returns:
        DataEfficiencyResults containing probe performance metrics for each dataset size
    """
    run_results = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Set seed for reproducibility
        pl.seed_everything(seed=run)

        # Convert the hyperopt config to baseline config
        baseline_config = get_finetune_config(config)

        # For each dataset size, subsample the train dataset and train the finetune baseline:
        finetune_baseline = FinetunedClassifier(baseline_config)

        # Train the finetune baseline:
        finetune_baseline.train(
            train_dataset,
            val_dataset=val_dataset,
        )

        eval_dataset_aurocs = []
        eval_dataset_accuracies = []
        eval_dataset_tpr_at_fprs = []

        for eval_dataset in tqdm(eval_datasets, desc="Eval Datasets", leave=False):
            results = finetune_baseline.get_results(eval_dataset)
            eval_dataset_aurocs.append(results.auroc())
            eval_dataset_accuracies.append(results.accuracy())
            eval_dataset_tpr_at_fprs.append(results.tpr_at_fixed_fpr(fpr=0.01)[0])

        # Calculate the metrics here:
        metrics = {
            "auroc": float(np.mean(eval_dataset_aurocs)),
            "accuracy": float(np.mean(eval_dataset_accuracies)),
            "tpr_at_fpr": float(np.mean(eval_dataset_tpr_at_fprs)),
        }

        run_results.append(metrics)

    # Calculate the mean and standard deviation across runs
    auroc_values = [run["auroc"] for run in run_results]
    accuracy_values = [run["accuracy"] for run in run_results]
    tpr_at_fpr_values = [run["tpr_at_fpr"] for run in run_results]

    result = FinetuneHyperOptResults(
        params={},  # Will be filled in by the caller
        auroc=float(np.mean(auroc_values)),
        accuracy=float(np.mean(accuracy_values)),
        tpr_at_fpr=float(np.mean(tpr_at_fpr_values)),
        auroc_std=float(np.std(auroc_values)),
        accuracy_std=float(np.std(accuracy_values)),
        tpr_at_fpr_std=float(np.std(tpr_at_fpr_values)),
        runs=run_results,
    )

    return result


def grid_search(
    base_config: FinetuneHyperConfig,
    param_grid: Dict[str, List[Any]],
    num_runs_per_config: int = 3,
) -> List[FinetuneHyperOptResults]:
    """Run a grid search over the hyperparameters.

    Args:
        base_config: Base configuration for the hyperopt experiment
        param_grid: Dictionary mapping parameter names to lists of values to try
        num_runs_per_config: Number of runs for each configuration to calculate variance

    Returns:
        List of results for each hyperparameter combination
    """
    # Generate all possible combinations of hyperparameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    results = []

    # Load datasets once to avoid reloading for each configuration
    train_dataset, val_dataset, eval_datasets = load_hyperopt_datasets(base_config)

    # Track the best configuration
    best_auroc = 0.0
    best_config = None

    for i, combination in enumerate(tqdm(param_combinations, desc="Grid Search")):
        # Create a config with the current parameter combination
        config_dict = base_config.model_dump()
        param_dict = {}

        for name, value in zip(param_names, combination):
            # Handle nested parameters
            if "." in name:
                parts = name.split(".")
                if parts[0] not in config_dict:
                    config_dict[parts[0]] = {}

                current_dict = config_dict
                for part in parts[:-1]:
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
                current_dict[parts[-1]] = value
            else:
                config_dict[name] = value

            # Store the parameter for results tracking
            param_dict[name] = value

        # Create the new config
        config = FinetuneHyperConfig(**config_dict)

        print(f"\nRunning configuration {i + 1}/{len(param_combinations)}:")
        for name, value in param_dict.items():
            print(f"  {name}: {value}")

        # Run the model with this configuration
        result = run_finetune_baseline(
            train_dataset,
            val_dataset,
            eval_datasets,
            config,
            num_runs=num_runs_per_config,
        )

        # Add the parameters to the result
        result.params = param_dict
        results.append(result)

        # Track best configuration (still using mean AUROC for selection)
        if result.auroc > best_auroc:
            best_auroc = result.auroc
            best_config = param_dict

        print(
            f"Results: AUROC={result.auroc:.4f}±{result.auroc_std:.4f}, "
            f"Accuracy={result.accuracy:.4f}±{result.accuracy_std:.4f}, "
            f"TPR@FPR={result.tpr_at_fpr:.4f}±{result.tpr_at_fpr_std:.4f}"
        )
        if best_config == param_dict:
            print("^ New best configuration!")

    # Before the best configuration, print the results for all configurations
    print("\nAll configurations:")
    for result in results:
        print(
            f"AUROC: {result.auroc:.4f}±{result.auroc_std:.4f}, "
            f"Accuracy: {result.accuracy:.4f}±{result.accuracy_std:.4f}, "
            f"TPR@FPR: {result.tpr_at_fpr:.4f}±{result.tpr_at_fpr_std:.4f}"
        )

    # Report the best configuration
    print("\nBest configuration:")
    assert best_config is not None, "No best configuration found"
    for name, value in best_config.items():
        print(f"  {name}: {value}")
    print(f"Best AUROC: {best_auroc:.4f}")

    return results


def run_hyperopt_experiment(
    config: FinetuneHyperConfig,
    param_grid: Dict[str, List[Any]],
    num_runs_per_config: int = 3,
) -> List[FinetuneHyperOptResults]:
    """Run a hyperparameter optimization experiment.

    Args:
        config: Base configuration for the hyperopt experiment
        param_grid: Dictionary mapping parameter names to lists of values to try
        num_runs_per_config: Number of runs for each configuration to calculate variance

    Returns:
        List of results for each hyperparameter combination
    """
    return grid_search(config, param_grid, num_runs_per_config)


if __name__ == "__main__":
    # Example usage
    base_config = FinetuneHyperConfig(
        model_name=LOCAL_MODELS["llama-70b"],  # Replace with your model name
        layer=31,
        dataset_path=Path(SYNTHETIC_DATASET_PATH),  # Replace with your dataset path
        compute_activations=False,
        eval_dataset_paths=list(
            EVAL_DATASETS_BALANCED.values()
        ),  # Replace with your eval dataset paths
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        num_classes=2,
        ClassifierModule={
            "learning_rate": 1e-5,
            "weight_decay": 0.1,
        },
        batch_size=4,
        shuffle=True,
        logger={
            "_target_": "pytorch_lightning.loggers.WandbLogger",
            "project": "models-under-pressure",
        },
        Trainer={
            "max_epochs": 30,
            "accelerator": "gpu",
            "devices": [0],
            "precision": "bf16-true",
            "default_root_dir": global_settings.PL_DEFAULT_ROOT_DIR,
            "accumulate_grad_batches": 4,
        },
    )

    # Define the parameter grid
    param_grid = {
        "batch_size": [8, 16, 32],
        "ClassifierModule.learning_rate": [1e-6, 1e-5, 1e-4],
        "ClassifierModule.weight_decay": [0.01, 1.0, 10.0],
    }

    # Run the grid search with 3 runs per configuration
    results = run_hyperopt_experiment(base_config, param_grid, num_runs_per_config=3)

from pathlib import Path
from typing import List, Optional

from models_under_pressure.config import (
    EVAL_DATASETS,
    SYNTHETIC_DATASET_PATH,
    DataEfficiencyBaselineConfig,
)
from models_under_pressure.dataset_utils import load_dataset, load_train_test
from models_under_pressure.finetune_baselines import FinetunedClassifier


def run_data_efficiency_finetune_baseline_with_activations(
    finetune_config: DataEfficiencyBaselineConfig,
    train_dataset_path: Path,
    eval_dataset_paths: List[Path],
    max_samples: Optional[int] = None,
) -> dict[Path, dict[str, float]]:
    print("Loading train dataset")
    train_dataset, val_dataset = load_train_test(
        dataset_path=train_dataset_path,
        model_name=None,
        layer=None,
        compute_activations=True,
        n_per_class=max_samples // 2 if max_samples else None,
    )
    print("Loading eval datasets")
    finetune_results = {}
    for eval_dataset_path in eval_dataset_paths:
        eval_dataset = load_dataset(
            dataset_path=eval_dataset_path,
            model_name=None,
            layer=None,
            compute_activations=True,
            n_per_class=max_samples // 2 if max_samples else None,
        )

        finetune_baseline = FinetunedClassifier(finetune_config)

        # Train the finetune baseline:
        finetune_baseline.train(
            train_dataset,
            val_dataset=val_dataset,
        )

        results = finetune_baseline.get_results(eval_dataset)

        # Calculate the metrics here:
        metrics = {
            "auroc": results.auroc(),
            "accuracy": results.accuracy(),
            "tpr_at_fpr": results.tpr_at_fixed_fpr(fpr=0.01)[0],
        }

        finetune_results[eval_dataset_path] = metrics

    # Incorporate the baseline results into the config:
    # results = DataEfficiencyResults(
    #    config=config,
    #    baseline_config=finetune_config,
    # )

    # return results
    return finetune_results


def ensure_wandb_login():
    """
    Ensures wandb is logged in by checking login status and prompting if needed.
    """
    import wandb

    try:
        # Check if already logged in
        wandb.ensure_configured()
        if wandb.api.api_key is None:
            print("WandB not logged in. Please enter your API key to log in.")
            wandb.login()
    except Exception as e:
        print(f"Error checking wandb login status: {e}")
        print("Please run 'wandb login' manually if you want to use wandb logging")


if __name__ == "__main__":
    ensure_wandb_login()

    # Should be defined via a hydra run config file:
    finetune_config = DataEfficiencyBaselineConfig(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        num_classes=2,
        ClassifierModule={  # set here to the default values
            "learning_rate": 1e-5,
            "weight_decay": 1.0,
            "scheduler_params": None,
            "class_weights": None,
            "label_smoothing": 0.0,
        },
        batch_size=4,
        shuffle=True,
        logger={
            "_target_": "pytorch_lightning.loggers.WandbLogger",
            "project": "models-under-pressure",
        },
        Trainer={
            "max_epochs": 1,  # 20,
            "accelerator": "gpu",
            "devices": [0],
            "precision": "bf16-true",
            # "default_root_dir": "/home/ubuntu/models-under-pressure/.cache",
            "default_root_dir": "/Users/john/code/models-under-pressure/.cache",
            "accumulate_grad_batches": 4,
        },
    )

    baseline_results = run_data_efficiency_finetune_baseline_with_activations(
        finetune_config,
        train_dataset_path=SYNTHETIC_DATASET_PATH,
        eval_dataset_paths=list(EVAL_DATASETS.values())[:2],
        max_samples=10,
    )

    # # Reload the results as a test:
    # with open(RESULTS_DIR / f"data_efficiency/results_{config.id}.jsonl", "r") as f:
    #     results_dict = json.loads(f.readlines()[-1])

    # results = DataEfficiencyResults.model_validate(results_dict)

import os
from pathlib import Path

import torch
from tqdm import tqdm

import wandb
from models_under_pressure.baselines.finetune import FinetunedClassifier
from models_under_pressure.config import (
    RESULTS_DIR,
    DataEfficiencyConfig,
    FinetuneBaselineConfig,
    global_settings,
)
from models_under_pressure.dataset_utils import load_dataset, load_train_test
from models_under_pressure.experiments.evaluate_probes import calculate_metrics
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.interfaces.probes import ProbeSpec, ProbeType
from models_under_pressure.interfaces.results import (
    DataEfficiencyBaselineResults,
    DatasetResults,
    FinetunedBaselineDataEfficiencyResults,
    ProbeDataEfficiencyResult,
)
from models_under_pressure.probes.base import Probe
from models_under_pressure.probes.probe_factory import ProbeFactory


def evaluate_probe(
    config: DataEfficiencyConfig,
    probe_spec: ProbeSpec,
    probe: Probe,
    eval_dataset_name: str,
    eval_dataset_path: Path,
    eval_dataset: LabelledDataset,
    output_dir: Path,
    train_dataset_size: int,
    fpr: float = 0.01,
) -> ProbeDataEfficiencyResult:
    """
    Evaluate a probe and save the results to a file.

    Args:
        probe: The probe to evaluate.
        eval_dataset_name: The name of the dataset to evaluate the probe on.
        eval_dataset: The dataset to evaluate the probe on.
        output_dir: The directory to save the results to.
        fpr: The FPR threshold to evaluate the probe at.
    Returns:
        The results of the probe evaluation.

    Method designed to be used in the data_efficiency.py experiment run
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # We want to evaluate the probe on the eval dataset
    per_entry_probe_scores = probe.predict_proba(eval_dataset)

    raw_metrics = calculate_metrics(
        eval_dataset.labels_numpy(),
        per_entry_probe_scores,
        fpr,
    )

    metrics = DatasetResults(
        layer=config.layer,
        metrics=raw_metrics,
    )

    try:
        best_epoch = probe._classifier.best_epoch  # type: ignore
    except AttributeError:
        best_epoch = None

    # Save the results into a DatasetResults instance
    results = ProbeDataEfficiencyResult(
        config=config,
        dataset_name=eval_dataset_name,
        dataset_path=eval_dataset_path,
        train_dataset_size=train_dataset_size,
        metrics=metrics,
        method=probe_spec.name,
        best_epoch=best_epoch,
        output_scores=per_entry_probe_scores.tolist(),
        output_labels=((per_entry_probe_scores > 0.5).astype(int)).tolist(),
        ground_truth_labels=eval_dataset.labels_numpy().tolist(),
        ground_truth_scale_labels=None,
        token_counts=None,
        ids=list(eval_dataset.ids),
        mean_of_masked_activations=None,
        masked_activations=None,
        per_token_probe_scores=None,
    )

    return results


def run_data_efficiency_experiment(
    config: DataEfficiencyConfig,
) -> None:
    """Run data efficiency experiment by training probes on different sized subsets of the dataset.

    Args:
        config: Configuration for the data efficiency experiment

    Returns:
        DataEfficiencyResults containing probe performance metrics for each dataset size
    """
    print("Loading train dataset")
    train_dataset, val_dataset = load_train_test(
        dataset_path=config.dataset_path,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )
    print("Loading eval datasets")
    eval_datasets = []
    eval_dataset_names = []
    _eval_dataset_paths = []
    for eval_dataset_name, eval_dataset_path in config.eval_dataset_paths.items():
        eval_datasets.append(
            load_dataset(
                dataset_path=eval_dataset_path,
                model_name=config.model_name,
                layer=config.layer,
                compute_activations=config.compute_activations,
            )
        )
        eval_dataset_names.append(eval_dataset_name)
        _eval_dataset_paths.append(eval_dataset_path)

    for dataset_size in tqdm(config.dataset_sizes, desc="Dataset sizes"):
        subset = subsample_balanced_subset(train_dataset, n_per_class=dataset_size // 2)

        for probe_spec in tqdm(config.probes, desc="Probes", leave=False):
            probe = ProbeFactory.build(
                probe_spec=probe_spec,
                train_dataset=subset,
                validation_dataset=val_dataset,
                model_name=config.model_name,
                layer=config.layer,
            )

            # Save the individual result to disk
            # Create a descriptive filename
            model_name_short = config.model_name.split("/")[-1]
            probe_name = probe_spec.name.value
            save_file = (
                config.results_dir
                / f"{model_name_short}_{probe_name}_probe_{dataset_size}.jsonl"
            )

            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_file.parent), exist_ok=True)

            # Delete prior content of the file
            if os.path.exists(save_file):
                os.remove(save_file)

            # Create a new empty file for the results
            if not os.path.exists(save_file):
                with open(save_file, "w") as f:
                    pass

            # For each dataset in eval_datasets, evaluate the probe and save the results:
            for i, (eval_dataset, eval_dataset_name) in enumerate(
                zip(eval_datasets, eval_dataset_names)
            ):
                probe_eval_results = evaluate_probe(
                    config=config,
                    probe_spec=probe_spec,
                    probe=probe,
                    eval_dataset_name=eval_dataset_name,
                    eval_dataset_path=_eval_dataset_paths[i],
                    eval_dataset=eval_dataset,
                    output_dir=config.results_dir,
                    train_dataset_size=dataset_size,
                    fpr=0.01,
                )

                print(
                    f"Saving probe results for {probe_name}, {eval_dataset_name} (size {dataset_size}) to {save_file}"
                )
                with open(save_file, "a") as f:
                    f.write(probe_eval_results.model_dump_json() + "\n")

            del probe
        del subset
        torch.cuda.empty_cache()


def run_data_efficiency_finetune_baseline_with_activations(
    config: DataEfficiencyConfig,
    finetune_config: FinetuneBaselineConfig,
) -> DataEfficiencyBaselineResults:
    """Run data efficiency experiment by training a finetuned baseline on different sized subsets of the dataset.

    Returns:
        DataEfficiencyResults containing probe performance metrics for each dataset size
    """

    print("Loading train dataset")
    train_dataset, val_dataset = load_train_test(
        dataset_path=config.dataset_path,
        model_name=None,
        layer=None,
        compute_activations=True,
    )

    print("Loading eval datasets")
    eval_datasets = []
    eval_dataset_names = []
    for eval_dataset_name, eval_dataset_path in config.eval_dataset_paths.items():
        eval_datasets.append(
            load_dataset(
                dataset_path=eval_dataset_path,
                model_name=None,
                layer=None,
                compute_activations=True,
            )
        )
        eval_dataset_names.append(eval_dataset_name)
    finetuned_baseline_results = []

    for dataset_size in tqdm(config.dataset_sizes, desc="Dataset sizes"):
        # For each dataset size, subsample the train dataset and train the finetune baseline:
        subset = subsample_balanced_subset(train_dataset, n_per_class=dataset_size // 2)
        finetune_baseline = FinetunedClassifier(finetune_config)

        # Train the finetune baseline:
        finetune_baseline.train(
            subset,
            val_dataset=val_dataset,
        )

        # For each eval dataset, get the full results using the finetune api:
        for eval_dataset, eval_dataset_name in tqdm(
            zip(eval_datasets, eval_dataset_names),
            desc=f"Evaluating datasets (size {dataset_size})",
            total=len(eval_datasets),
        ):
            results = finetune_baseline.get_full_results(
                eval_dataset=eval_dataset,
                dataset_name=eval_dataset_name,
                eval_dataset_path=config.eval_dataset_paths[eval_dataset_name],
            )

            # Convert the results to a FinetunedBaselineDataEfficiencyResults instance:
            finetuned_baseline_results.append(
                FinetunedBaselineDataEfficiencyResults.from_finetuned_baseline_results(
                    results, dataset_size
                )
            )

            # Create a more descriptive filename for saving results
            model_name_short = finetune_config.model_name_or_path.split("/")[-1]
            save_file = (
                config.results_dir / f"{model_name_short}_baseline_{dataset_size}.jsonl"
            )

            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_file.parent), exist_ok=True)
            # Create empty file if it doesn't exist
            if not os.path.exists(save_file):
                open(save_file, "w").close()
            print(f"Saving results for {eval_dataset_name} to {save_file}")

            # with open(save_file, "a") as f:
            #     f.write(finetuned_baseline_results[-1].model_dump_json() + "\n")

            finetuned_baseline_results[-1].save_to(save_file)

        # After each eval, clear the cache, delete the baseline model and dataset subset.
        torch.cuda.empty_cache()
        del finetune_baseline
        del subset

    # Incorporate the baseline results into the config:
    results = DataEfficiencyBaselineResults(
        config=config,  # Main experiment config
        baseline_config=finetune_config,
        baseline_results=finetuned_baseline_results,
    )

    return results


def ensure_wandb_login():
    """
    Ensures wandb is logged in by checking login status and prompting if needed.
    """

    try:
        # Check if already logged in
        wandb.ensure_configured()  # type: ignore
        if wandb.api.api_key is None:
            print("WandB not logged in. Please enter your API key to log in.")
            wandb.login()
    except Exception as e:
        print(f"Error checking wandb login status: {e}")
        print("Please run 'wandb login' manually if you want to use wandb logging")


if __name__ == "__main__":
    from models_under_pressure.config import (
        EVAL_DATASETS_BALANCED,
        LOCAL_MODELS,
        SYNTHETIC_DATASET_PATH,
        FinetuneBaselineConfig,
    )

    ensure_wandb_login()

    config = DataEfficiencyConfig(
        model_name=LOCAL_MODELS["llama-70b"],
        layer=31,
        dataset_path=SYNTHETIC_DATASET_PATH,
        dataset_sizes=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 1910],
        probes=[
            ProbeSpec(
                name=ProbeType.sklearn,
                hyperparams={"C": 1e-3, "random_state": 42, "fit_intercept": False},
            ),
            ProbeSpec(
                name=ProbeType.attention,
                hyperparams={
                    "name": "attention",
                    "batch_size": 16,
                    "epochs": 200,
                    "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                    "final_lr": 5e-4,
                    "gradient_accumulation_steps": 4,
                    "patience": 50,
                },
            ),
            ProbeSpec(
                name=ProbeType.linear_then_softmax,
                hyperparams={
                    "temperature": 5,
                    "batch_size": 16,
                    "epochs": 200,
                    "optimizer_args": {"lr": 5e-3, "weight_decay": 1e-3},
                    "final_lr": 1e-4,
                    "gradient_accumulation_steps": 4,
                    "patience": 10,
                },
            ),
        ],
        compute_activations=False,
        eval_dataset_paths=EVAL_DATASETS_BALANCED,
        results_dir=RESULTS_DIR / "data_efficiency",
        # id="g6AooBhS",
    )

    # Should be defined via a hydra run config file:
    finetune_config = FinetuneBaselineConfig(
        model_name_or_path=LOCAL_MODELS["llama-8b"],
        num_classes=2,
        ClassifierModule={  # set here to the default values
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "scheduler_params": None,
            "class_weights": None,
            "label_smoothing": 0.0,
            "optimizer": "adamw8bit",
        },
        batch_size=2,
        test_batch_size=2,
        shuffle=True,
        num_workers=10,
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

    # results = run_data_efficiency_experiment(config)
    baseline_results = run_data_efficiency_finetune_baseline_with_activations(
        config, finetune_config
    )

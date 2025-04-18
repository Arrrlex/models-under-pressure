import json

import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from models_under_pressure.config import RESULTS_DIR, DataEfficiencyConfig
from models_under_pressure.dataset_utils import load_dataset, load_train_test
from models_under_pressure.finetune_baselines import FinetunedClassifier
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.interfaces.results import (
    DataEfficiencyResults,
    ProbeDataEfficiencyResults,
)
from models_under_pressure.probes.base import Probe
from models_under_pressure.probes.metrics import tpr_at_fixed_fpr_score
from models_under_pressure.probes.probes import ProbeFactory


def evaluate_probe(
    probe: Probe, eval_datasets: list[LabelledDataset]
) -> dict[str, float]:
    """Evaluate a probe on a list of datasets.

    Args:
        probe: The probe to evaluate
        eval_datasets: A list of datasets to evaluate the probe on
    """
    probe_scores = np.concatenate(
        [probe.predict_proba(ds)[1] for ds in eval_datasets], axis=0
    )
    labels = np.concatenate([ds.labels_numpy() for ds in eval_datasets], axis=0)
    return {
        "auroc": float(roc_auc_score(labels, probe_scores)),
        "accuracy": float(accuracy_score(labels, probe_scores > 0.5)),
        "tpr_at_fpr": float(tpr_at_fixed_fpr_score(labels, probe_scores, fpr=0.01)),
    }


def run_data_efficiency_experiment(
    config: DataEfficiencyConfig,
) -> DataEfficiencyResults:
    """Run data efficiency experiment by training probes on different sized subsets of the dataset.

    Args:
        config: Configuration for the data efficiency experiment

    Returns:
        DataEfficiencyResults containing probe performance metrics for each dataset size
    """
    print("Loading train dataset")
    train_dataset, _ = load_train_test(
        dataset_path=config.dataset_path,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )
    print("Loading eval datasets")
    eval_datasets = []
    for eval_dataset_path in config.eval_dataset_paths:
        eval_datasets.append(
            load_dataset(
                dataset_path=eval_dataset_path,
                model_name=config.model_name,
                layer=config.layer,
                compute_activations=config.compute_activations,
            )
        )

    probe_results = []

    for dataset_size in tqdm(config.dataset_sizes, desc="Dataset sizes"):
        subset = subsample_balanced_subset(train_dataset, n_per_class=dataset_size // 2)

        for probe_spec in tqdm(config.probes, desc="Probes", leave=False):
            probe = ProbeFactory.build(probe=probe_spec, train_dataset=subset)
            metrics = evaluate_probe(probe, eval_datasets)

            probe_results.append(
                ProbeDataEfficiencyResults(
                    probe=probe_spec,
                    dataset_size=dataset_size,
                    metrics=metrics,
                )
            )

    results = DataEfficiencyResults(
        config=config,
        probe_results=probe_results,
    )

    results.save_to(config.output_path)

    return results


def run_data_efficiency_finetune_baseline_with_activations(
    config: DataEfficiencyConfig,
    finetune_config: DictConfig,
) -> DataEfficiencyResults:
    """Run data efficiency experiment by training probes on different sized subsets of the dataset.

    Args:
        config: Configuration for the data efficiency experiment

    Returns:
        DataEfficiencyResults containing probe performance metrics for each dataset size
    """

    print("Loading train dataset")
    train_dataset, _ = load_train_test(
        dataset_path=config.dataset_path,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )
    print("Loading eval datasets")
    eval_datasets = []
    for eval_dataset_path in config.eval_dataset_paths:
        eval_datasets.append(
            load_dataset(
                dataset_path=eval_dataset_path,
                model_name=config.model_name,
                layer=config.layer,
                compute_activations=config.compute_activations,
            )
        )

    probe_results = []

    for dataset_size in tqdm(config.dataset_sizes, desc="Dataset sizes"):
        subset = subsample_balanced_subset(train_dataset, n_per_class=dataset_size // 2)

        for probe_spec in tqdm(config.probes, desc="Probes", leave=False):
            finetune_baseline = FinetunedClassifier(finetune_config)

            # Train the finetune baseline:
            finetune_baseline.train(subset, train_dataset.labels_numpy())

            # Predict the probability of each example in the dataset being high-stakes:
            # probits = finetune_baseline.predict_proba(subset)

            # Calculate the metrics here:

            probe_results.append(
                ProbeDataEfficiencyResults(
                    probe=probe_spec,
                    dataset_size=dataset_size,
                    metrics={},
                )
            )

    results = DataEfficiencyResults(
        config=config,
        probe_results=probe_results,
    )

    results.save_to(config.output_path)

    return results


def plot_data_efficiency_results(results: DataEfficiencyResults, metric: str = "auroc"):
    """Plot probe performance vs dataset size.

    Args:
        results: DataEfficiencyResults containing probe performance data
        metric: Which metric to plot. One of "auroc", "accuracy", or "tpr_at_fpr"
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    sns.set_theme(style="whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group results by probe type
    probe_types = {probe.name for probe in results.config.probes}

    # Plot line for each probe type
    for probe_type in probe_types:
        # Get results for this probe type
        probe_results = [r for r in results.probe_results if r.probe.name == probe_type]

        # Sort by dataset size
        probe_results.sort(key=lambda x: x.dataset_size)

        # Extract x and y values
        x = [r.dataset_size for r in probe_results]
        y = [r.metrics[metric] for r in probe_results]

        # Plot line
        ax.plot(x, y, marker="o", label=probe_type)

    # Set labels and title
    ax.set_xlabel("Dataset Size (samples)")
    metric_labels = {
        "auroc": "AUROC",
        "accuracy": "Accuracy",
        "tpr_at_fpr": "TPR at 1% FPR",
    }
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(f"Probe Performance vs Dataset Size\n{results.config.model_name}")

    # Use log scale for x-axis since dataset sizes often vary by orders of magnitude
    ax.set_xscale("log")

    # Add grid
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Add legend
    ax.legend(title="Probe Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir = Path(results.config.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save plots
    base_path = output_dir / f"data_efficiency_{results.config.id}_{metric}"
    plt.savefig(f"{base_path}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)

    return fig


if __name__ == "__main__":
    from models_under_pressure.config import (
        EVAL_DATASETS_BALANCED,
        LOCAL_MODELS,
        SYNTHETIC_DATASET_PATH,
    )
    from models_under_pressure.interfaces.probes import ProbeSpec

    config = DataEfficiencyConfig(
        model_name=LOCAL_MODELS["llama-1b"],
        layer=11,
        dataset_path=SYNTHETIC_DATASET_PATH,
        dataset_sizes=[2, 4, 8, 16, 32, 64, 128, 256, 512, 836],
        probes=[
            ProbeSpec(
                name="sklearn_mean_agg_probe",
                hyperparams={"C": 1e-3, "random_state": 42, "fit_intercept": False},
            ),
            ProbeSpec(
                name="pytorch_per_token_probe",
                hyperparams={
                    "batch_size": 16,
                    "epochs": 3,
                    "device": "cuda",
                    "learning_rate": 1e-2,
                    "weight_decay": 0.1,
                },
            ),
            ProbeSpec(
                name="difference_of_means",
                hyperparams={"batch_size": 16, "epochs": 3, "device": "cpu"},
            ),
        ],
        compute_activations=True,
        eval_dataset_paths=list(EVAL_DATASETS_BALANCED.values()),
        # id="g6AooBhS",
    )
    results = run_data_efficiency_experiment(config)

    with open(RESULTS_DIR / f"data_efficiency/results_{config.id}.jsonl", "r") as f:
        results_dict = json.loads(f.readlines()[-1])

    results = DataEfficiencyResults.model_validate(results_dict)
    plot_data_efficiency_results(results)

from models_under_pressure.config import DataEfficiencyConfig
from models_under_pressure.interfaces.dataset import subsample_balanced_subset
from models_under_pressure.interfaces.results import (
    DataEfficiencyResults,
    ProbeDataEfficiencyResults,
)
from models_under_pressure.dataset_utils import load_train_test
from models_under_pressure.probes.probes import ProbeFactory
from models_under_pressure.probes.metrics import tpr_at_fixed_fpr_score
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


def run_data_efficiency_experiment(
    config: DataEfficiencyConfig,
) -> DataEfficiencyResults:
    """Run data efficiency experiment by training probes on different sized subsets of the dataset.

    Args:
        config: Configuration for the data efficiency experiment

    Returns:
        DataEfficiencyResults containing probe performance metrics for each dataset size
    """
    # Load the full dataset
    train_dataset, _ = load_train_test(
        dataset_path=config.dataset_path,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )

    probe_results = []

    # For each dataset size
    for dataset_size in tqdm(config.dataset_sizes, desc="Dataset sizes"):
        # Create a subset of the dataset
        subset = subsample_balanced_subset(train_dataset, n_per_class=dataset_size // 2)

        # For each probe type
        for probe_spec in tqdm(config.probes, desc="Probes", leave=False):
            # Train the probe on the subset
            probe = ProbeFactory.build(probe=probe_spec, train_dataset=subset)

            # Evaluate the probe on the full dataset
            _, pred_scores = probe.predict_proba(train_dataset)
            pred_labels = pred_scores > 0.5
            labels = train_dataset.labels_numpy()

            # Calculate metrics
            metrics = {
                "accuracy": float(accuracy_score(labels, pred_labels)),
                "auroc": float(roc_auc_score(labels, pred_scores)),
                "tpr_at_fpr": float(
                    tpr_at_fixed_fpr_score(labels, pred_scores, fpr=0.01)
                ),
            }

            # Store results
            probe_results.append(
                ProbeDataEfficiencyResults(
                    probe=probe_spec,
                    dataset_size=dataset_size,
                    metrics=metrics,
                )
            )

    # Create and return results
    results = DataEfficiencyResults(
        config=config,
        probe_results=probe_results,
    )

    # Save results
    results.save_to(config.output_path)

    return results


def plot_data_efficiency_results(results: DataEfficiencyResults, metric: str = "auroc"):
    """Plot probe performance vs dataset size.

    Args:
        results: DataEfficiencyResults containing probe performance data
        metric: Which metric to plot. One of "auroc", "accuracy", or "tpr_at_fpr"
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Set style
    plt.style.use("seaborn")
    sns.set_palette("husl")

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
    from models_under_pressure.config import SYNTHETIC_DATASET_PATH, LOCAL_MODELS
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
                    "device": "cpu",
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
    )
    results = run_data_efficiency_experiment(config)
    plot_data_efficiency_results(results)

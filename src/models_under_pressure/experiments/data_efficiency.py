from models_under_pressure.config import DataEfficiencyConfig
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
        subset = train_dataset.sample(dataset_size)

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


if __name__ == "__main__":
    from models_under_pressure.config import SYNTHETIC_DATASET_PATH
    from models_under_pressure.interfaces.probes import ProbeSpec

    config = DataEfficiencyConfig(
        model_name="llama-1b",
        layer=11,
        dataset_path=SYNTHETIC_DATASET_PATH,
        dataset_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1319],
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
    )
    run_data_efficiency_experiment(config)

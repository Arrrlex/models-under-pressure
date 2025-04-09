import json

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models_under_pressure.config import (
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
    VARIATION_TYPES,
    HeatmapRunConfig,
    ProbeSpec,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_train_test,
    split_by_variation,
)
from models_under_pressure.interfaces.results import (
    HeatmapCellResult,
    HeatmapRunResults,
)
from models_under_pressure.probes.probes import ProbeFactory
from models_under_pressure.utils import double_check_config, print_progress
from models_under_pressure.experiments.train_probes import tpr_at_fixed_fpr_score


def generate_heatmaps(config: HeatmapRunConfig) -> HeatmapRunResults:
    """Generate heatmaps for a given model and dataset.

    This function generates heatmaps for a given model and dataset by training
    a probe on a train set portion with a single variation value and evaluating
    on all test set portions with various variation values.
    """
    train_dataset, test_dataset = load_train_test(
        dataset_path=config.dataset_path,
        model_name=config.model_name,
        layer=config.layer,
    )

    results_list = []

    for variation_type in print_progress(config.variation_types):
        print(f"\nGenerating heatmap for {variation_type}")

        variations = split_by_variation(
            train_dataset,
            test_dataset,
            variation_type,
            max_samples=config.max_samples,
        )

        for train_variation_value in tqdm(variations.variation_values):
            print(f"Training on variation '{variation_type}'='{train_variation_value}'")
            train_split = variations.train_splits[train_variation_value]
            probe = ProbeFactory.build(
                probe=config.probe_spec,
                train_dataset=train_split,
            )

            for test_variation_value in variations.variation_values:
                print(
                    f"Evaluating on variation '{variation_type}'='{test_variation_value}'"
                )
                test_split = variations.test_splits[test_variation_value]
                _, pred_scores = probe.predict_proba(test_split)
                pred_labels = pred_scores > 0.5
                labels = test_split.labels_numpy()
                metrics = {
                    "accuracy": (pred_labels == labels).mean(),
                    "tpr_at_1pct_fpr": tpr_at_fixed_fpr_score(
                        y_true=labels, y_pred=pred_scores, fpr=0.01
                    ),
                    "auroc": roc_auc_score(labels, pred_scores),
                }

                result = HeatmapCellResult(
                    variation_type=variation_type,
                    train_variation_value=train_variation_value,
                    test_variation_value=test_variation_value,
                    metrics=metrics,
                )

                print(result)

                results_list.append(result)

                with open(config.intermediate_output_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": config.id,
                                "timestamp": config.timestamp.isoformat(),
                                "result": result.model_dump(mode="json"),
                            }
                        )
                        + "\n"
                    )

    results = HeatmapRunResults(
        config=config,
        results=results_list,
    )

    print(f"\nSaving final heatmap results to {config.output_path}")
    with open(config.output_path, "a") as f:
        f.write(results.model_dump_json() + "\n")
    print(f"Done! Results saved to {config.output_path}")

    return results


if __name__ == "__main__":
    config = HeatmapRunConfig(
        layer=11,
        max_samples=None,
        model_name=LOCAL_MODELS["llama-1b"],
        dataset_path=SYNTHETIC_DATASET_PATH,
        variation_types=VARIATION_TYPES,
        probe_spec=ProbeSpec(name="sklearn_mean_agg_probe"),
    )

    double_check_config(config)

    generate_heatmaps(config)

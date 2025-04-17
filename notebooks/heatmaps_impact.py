import itertools
import json

import numpy as np
from jaxtyping import Float
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from models_under_pressure.config import (
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
    VARIATION_TYPES,
    HeatmapRunConfig,
    ProbeSpec,
)
from models_under_pressure.dataset_utils import (
    load_train_test,
)
from models_under_pressure.interfaces.results import (
    HeatmapCellResult,
    HeatmapRunResults,
)
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probe_factory import ProbeFactory
from models_under_pressure.utils import double_check_config


def compute_tpr_at_1pct_fpr(
    pred_scores: Float[np.ndarray, " batch_size"],
    labels: Float[np.ndarray, " batch_size"],
) -> float:
    try:
        fpr, tpr, _ = roc_curve(labels, pred_scores)
        return float(tpr[np.where(fpr <= 0.01)[0][-1]])
    except Exception as e:
        print(f"Error computing TPR at 1% FPR: {e}")
        return -1.0


def generate_heatmaps(config: HeatmapRunConfig) -> HeatmapRunResults:
    """Generate heatmaps for a given model and dataset.

    This function generates heatmaps for a given model and dataset by training
    a probe on a train set portion with a single variation value and evaluating
    on all test set portions with various variation values.
    """
    train_dataset, test_dataset = load_train_test(
        dataset_path=config.dataset_path,
    )
    results_list = []
    model = LLMModel.load(config.model_name)
    # combine train and test datasets
    train_dataset = train_dataset.concatenate([test_dataset])
    # seperate train and test based on column - "impact factor" such that first, it trains on specific impact factor values and then tests on all other values
    # and then repeates this for combination of two and three impact factors
    # read impact factor from json
    impact_factors = list(set(train_dataset.other_fields["impact_factor"]))
    for num_impact_factors in tqdm(range(1, 4)):
        # create list of n combinations of impact factors
        impact_factors_combinations = list(
            itertools.combinations(impact_factors, num_impact_factors)
        )

        for i in tqdm(range(len(impact_factors_combinations))):
            impact_factors_train = impact_factors_combinations[i]
            train_dataset_impact_factor = train_dataset.filter(
                lambda x: x.other_fields["impact_factor"] in impact_factors_train
            )
            train_dataset_impact_factor = train_dataset_impact_factor.sample(
                min(config.max_samples, len(train_dataset_impact_factor))
                if config.max_samples is not None
                else len(train_dataset_impact_factor)
            )
            print(f"\nGenerating heatmap for {impact_factors_train}")

            print(f"Training on variation '{impact_factors_train}'")
            train_split = train_dataset_impact_factor
            probe = ProbeFactory.build(
                probe=config.probe_spec,
                model=model,
                train_dataset=train_split,
                layer=config.layer,
            )

            for j in tqdm(range(len(impact_factors_combinations))):
                test_impact_factors = impact_factors_combinations[j]
                test_dataset_impact_factor = train_dataset.filter(
                    lambda x: x.other_fields["impact_factor"] in test_impact_factors
                )
                test_dataset_impact_factor = test_dataset_impact_factor.sample(
                    min(config.max_samples, len(test_dataset_impact_factor))
                    if config.max_samples is not None
                    else len(test_dataset_impact_factor)
                )
                _, pred_scores = probe.predict_proba(test_dataset_impact_factor)
                pred_labels = pred_scores > 0.5
                labels = test_dataset_impact_factor.labels_numpy()
                metrics = {
                    "accuracy": (pred_labels == labels).mean(),
                    "tpr_at_1pct_fpr": compute_tpr_at_1pct_fpr(pred_scores, labels),
                    "auroc": roc_auc_score(labels, pred_scores),
                }

                result = HeatmapCellResult(
                    variation_type=f"{num_impact_factors}",
                    train_variation_value="_".join(impact_factors_train),
                    test_variation_value="_".join(test_impact_factors),
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
        id="impact_heatmap",
        layer=31,
        max_samples=None,
        model_name=LOCAL_MODELS["llama-70b"],
        dataset_path=SYNTHETIC_DATASET_PATH,
        variation_types=VARIATION_TYPES,
        probe_spec=ProbeSpec(name="sklearn_mean_agg_probe"),
    )

    double_check_config(config)

    generate_heatmaps(config)

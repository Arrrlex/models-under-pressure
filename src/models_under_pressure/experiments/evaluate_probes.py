# Code to generate Figure 2
import json

import numpy as np
from tqdm import tqdm

from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    TEST_DATASETS,
    EvalRunConfig,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.experiments.train_probes import (
    evaluate_probe_and_save_results,
    get_coefs,
)
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import EvaluationResult
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.probes import ProbeFactory
from models_under_pressure.utils import double_check_config


def run_evaluation(
    config: EvalRunConfig,
) -> tuple[list[EvaluationResult], list[float]]:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    train_dataset = load_filtered_train_dataset(
        dataset_path=config.dataset_path,
        variation_type=config.variation_type,
        variation_value=config.variation_value,
        max_samples=config.max_samples,
        model_name=config.model_name,
        layer=config.layer,
    )

    # Create the probe:
    print("Creating probe ...")
    probe = ProbeFactory.build(
        probe=config.probe_spec,
        train_dataset=train_dataset,
    )

    del train_dataset

    coefs = get_coefs(probe)

    eval_dataset_paths = TEST_DATASETS if config.use_test_set else EVAL_DATASETS

    results_list = []

    if config.compute_activations:
        model = LLMModel.load(config.model_name)

    for eval_dataset_name, eval_dataset_path in tqdm(
        eval_dataset_paths.items(), desc="Evaluating on eval datasets"
    ):
        print(f"Loading eval dataset {eval_dataset_name} from {eval_dataset_path}")
        if config.compute_activations:
            eval_dataset = LabelledDataset.load_from(eval_dataset_path)
            if config.max_samples and len(eval_dataset) > config.max_samples:
                eval_dataset = subsample_balanced_subset(
                    eval_dataset, n_per_class=config.max_samples // 2
                )
            print(f"Computing activations for {eval_dataset_name} ...")
            activations = model.get_batched_activations(
                eval_dataset, layer=config.layer
            )
            eval_dataset = eval_dataset.assign(
                activations=activations._activations,
                attention_mask=activations._attention_mask,
                input_ids=activations._input_ids,
            )
        else:
            eval_dataset = ActivationStore().load_enriched_dataset(
                dataset_path=eval_dataset_path,
                model_name=config.model_name,
                layer=config.layer,
            )

            if config.max_samples and len(eval_dataset) > config.max_samples:
                eval_dataset = subsample_balanced_subset(
                    eval_dataset, n_per_class=config.max_samples // 2
                )

        print(f"Evaluating probe on {eval_dataset_name} ...")
        probe_scores, dataset_results = evaluate_probe_and_save_results(
            probe=probe,
            train_dataset_path=config.dataset_path,
            eval_dataset_name=eval_dataset_name,
            eval_dataset=eval_dataset,
            model_name=config.model_name,
            layer=config.layer,
            output_dir=EVALUATE_PROBES_DIR,
        )

        ground_truth_labels = eval_dataset.labels_numpy().tolist()

        if "scale_labels" in eval_dataset.other_fields:
            ground_truth_scale_labels = [
                int(label) for label in eval_dataset.other_fields["scale_labels"]
            ]
        else:
            ground_truth_scale_labels = None

        print(f"Metrics for {eval_dataset_name}: {dataset_results.metrics}")

        dataset_results = EvaluationResult(
            config=config,
            metrics=dataset_results,
            dataset_name=eval_dataset_name,
            method="linear_probe",
            output_scores=probe_scores,
            output_labels=list(int(a > 0.5) for a in probe_scores),
            ground_truth_scale_labels=ground_truth_scale_labels,
            ground_truth_labels=ground_truth_labels,
            dataset_path=eval_dataset_path,
        )

        results_list.append(dataset_results)

        del eval_dataset

    return results_list, coefs


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        layer=11,
        max_samples=20,
        model_name=LOCAL_MODELS["llama-1b"],
        probe_spec=ProbeSpec(
            name="pytorch_per_token_probe",
            hyperparams={"batch_size": 16, "epochs": 3, "device": "cpu"},
        ),
        compute_activations=True,
    )

    double_check_config(config)

    print(f"Running probe evaluation with ID {config.id}")
    print(f"Results will be saved to {EVALUATE_PROBES_DIR / config.output_filename}")
    results, coefs = run_evaluation(config=config)

    print(f"Saving results to {EVALUATE_PROBES_DIR / config.output_filename}")
    for result in results:
        result.save_to(EVALUATE_PROBES_DIR / config.output_filename)

    coefs_dict = {
        "id": config.id,
        "coefs": coefs[0].tolist(),  # type: ignore
    }
    with open(EVALUATE_PROBES_DIR / config.coefs_filename, "w") as f:
        json.dump(coefs_dict, f)

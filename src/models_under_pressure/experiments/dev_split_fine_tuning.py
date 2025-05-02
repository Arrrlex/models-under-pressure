from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
    DevSplitFineTuningConfig,
)
from models_under_pressure.dataset_utils import load_dataset, load_splits_lazy
from models_under_pressure.experiments.evaluate_probes import calculate_metrics
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import DatasetResults, KShotResult
from models_under_pressure.probes.base import Probe
from models_under_pressure.probes.probe_factory import ProbeFactory
from models_under_pressure.utils import double_check_config


def evaluate_probe(
    probe: Probe,
    eval_dataset: LabelledDataset,
    fpr: float = 0.01,
) -> tuple[List[float], DatasetResults]:
    """Evaluate a probe on a dataset and return scores and metrics."""
    per_entry_probe_scores = probe.predict_proba(eval_dataset)
    y_true = eval_dataset.labels_numpy()
    y_pred = per_entry_probe_scores

    return (
        per_entry_probe_scores.tolist(),
        DatasetResults(layer=0, metrics=calculate_metrics(y_true, y_pred, fpr)),
    )


def run_dev_split_fine_tuning(config: DevSplitFineTuningConfig) -> List[KShotResult]:
    # Load and split the training dataset
    splits = load_splits_lazy(
        dataset_path=config.dataset_path,
        dataset_filters=None,
        n_per_class=config.max_samples,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )

    # Create initial probe
    print("Creating initial probe...")
    probe = ProbeFactory.build(
        probe=config.probe_spec,
        train_dataset=splits["train"],
        validation_dataset=splits["test"] if config.validation_dataset else None,
    )

    # Save the initial probe state
    if hasattr(probe, "_classifier") and hasattr(probe._classifier, "model"):
        initial_state = probe._classifier.model.state_dict().copy()
    else:
        initial_state = None

    results_list = []

    for eval_dataset_path in tqdm(
        config.eval_datasets, desc="Evaluating on eval datasets"
    ):
        eval_dataset_name = eval_dataset_path.stem
        print(f"Loading eval dataset {eval_dataset_name} from {eval_dataset_path}")
        eval_dataset = load_dataset(
            dataset_path=eval_dataset_path,
            model_name=config.model_name,
            layer=config.layer,
            compute_activations=config.compute_activations,
            n_per_class=config.max_samples // 2 if config.max_samples else None,
        )

        # Split eval dataset into train and test
        train_indices, test_indices = train_test_split(
            range(len(eval_dataset)),
            train_size=config.train_split_ratio,
            random_state=42,
            stratify=eval_dataset.labels_numpy(),
        )

        train_split = eval_dataset[train_indices]
        test_split = eval_dataset[test_indices]

        # Load initial probe state
        if initial_state is not None:
            probe._classifier.model.load_state_dict(initial_state)
        else:
            probe = ProbeFactory.build(
                probe=config.probe_spec,
                train_dataset=splits["train"],
                validation_dataset=splits["test"]
                if config.validation_dataset
                else None,
            )

        # Evaluate initial probe on test split
        initial_scores, initial_metrics = evaluate_probe(probe, test_split)
        initial_result = KShotResult(
            config=config,
            k=0,
            metrics=initial_metrics.metrics,
            probe_scores=initial_scores,
            ground_truth_labels=test_split.labels_numpy().tolist(),
            ground_truth_scale_labels=test_split.other_fields.get("scale_labels"),
            dataset_name=f"{eval_dataset_name}_k0",
            dataset_path=eval_dataset_path,
            method="initial_probe",
        )
        results_list.append(initial_result)

        # Fine-tune and evaluate for each k
        for k in tqdm(config.k_values, desc=f"Fine-tuning on {eval_dataset_name}"):
            if k > len(train_split):
                print(
                    f"Skipping k={k} as it's larger than train split size {len(train_split)}"
                )
                continue

            # Sample k examples from train split
            k_split = subsample_balanced_subset(train_split, n_per_class=k // 2)

            if config.eval_data_usage == "combine":
                # Combine train split and k_split
                # Get common fields between train_split and k_split
                train_fields = set(train_split.other_fields.keys())
                k_fields = set(k_split.other_fields.keys())
                common_fields = train_fields.intersection(k_fields)

                # Create new datasets with only common fields
                train_split_filtered = LabelledDataset(
                    inputs=train_split.inputs,
                    ids=train_split.ids,
                    other_fields={
                        k: train_split.other_fields[k] for k in common_fields
                    },
                )
                k_split_filtered = LabelledDataset(
                    inputs=k_split.inputs,
                    ids=k_split.ids,
                    other_fields={k: k_split.other_fields[k] for k in common_fields},
                )
                combined_split = LabelledDataset.concatenate(
                    [train_split_filtered] + [k_split_filtered] * config.sample_repeats
                )
                probe = ProbeFactory.build(
                    probe=config.probe_spec,
                    train_dataset=combined_split,
                    validation_dataset=None,
                )
            elif config.eval_data_usage == "only":
                probe = ProbeFactory.build(
                    probe=config.probe_spec,
                    train_dataset=k_split,
                    validation_dataset=None,
                )
            elif config.eval_data_usage == "fine-tune":
                # Restore initial probe state before fine-tuning
                if (
                    initial_state is not None
                    and hasattr(probe, "_classifier")
                    and hasattr(probe._classifier, "model")
                ):
                    probe._classifier.model.load_state_dict(initial_state)
                else:
                    raise NotImplementedError(
                        "Cannot restore initial probe state for this probe type"
                    )

                # Fine-tune probe
                print(f"Fine-tuning probe on {k} examples...")
                if hasattr(probe, "_classifier") and hasattr(
                    probe._classifier, "training_args"
                ):
                    probe._classifier.training_args["epochs"] = config.fine_tune_epochs
                probe.fit(k_split)
            else:
                raise ValueError(
                    f"Invalid eval_data_usage: {config.eval_data_usage}. Must be one of: 'fine-tune', 'only', 'combine'"
                )

            # Evaluate fine-tuned probe
            fine_tuned_scores, fine_tuned_metrics = evaluate_probe(probe, test_split)
            fine_tuned_result = KShotResult(
                config=config,
                k=k,
                metrics=fine_tuned_metrics.metrics,
                probe_scores=fine_tuned_scores,
                ground_truth_labels=test_split.labels_numpy().tolist(),
                ground_truth_scale_labels=test_split.other_fields.get("scale_labels"),
                dataset_name=f"{eval_dataset_name}_k{k}",
                dataset_path=eval_dataset_path,
                method="fine_tuned_probe",
            )
            results_list.append(fine_tuned_result)

    # Save results
    print(f"Saving results to {EVALUATE_PROBES_DIR / config.output_filename}")
    for result in results_list:
        result.save_to(EVALUATE_PROBES_DIR / config.output_filename)

    return results_list


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    config = DevSplitFineTuningConfig(
        layer=31,
        max_samples=None,
        # fine_tune_epochs=10,
        eval_data_usage="only",
        model_name=LOCAL_MODELS["llama-70b"],
        probe_spec=ProbeSpec(
            name="sklearn_mean_agg_probe",
            hyperparams={"C": 1e-3, "fit_intercept": False},
            # name="pytorch_per_entry_probe_mean",
            # hyperparams={
            #     "batch_size": 16,
            #     "epochs": 20,
            #     "device": "cpu",
            #     "optimizer_args": {"lr": 0.001, "weight_decay": 0.01},
            # },
        ),
        compute_activations=False,
        dataset_path=SYNTHETIC_DATASET_PATH,
        validation_dataset=False,
        # eval_datasets=[EVAL_DATASETS["anthropic"]],
        eval_datasets=list(EVAL_DATASETS.values()),
    )

    double_check_config(config)

    for k in range(4):
        print("Running k-shot fine-tuning experiment")
        print(
            f"Results will be saved to {EVALUATE_PROBES_DIR / config.output_filename}"
        )
        results = run_dev_split_fine_tuning(config)
        for result in results:
            print("-" * 100)
            print(result.dataset_name)
            print(result.metrics)

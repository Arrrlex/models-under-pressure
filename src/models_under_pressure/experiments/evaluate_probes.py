# Code to generate Figure 2
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path

import numpy as np
from sklearn.base import accuracy_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models_under_pressure.config import (
    EVALUATE_PROBES_DIR,
    EvalRunConfig,
)
from models_under_pressure.dataset_utils import (
    load_train_test,
)
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
)
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import EvaluationResult
from models_under_pressure.interfaces.results import DatasetResults
from models_under_pressure.probes.base import Probe
from models_under_pressure.probes.metrics import tpr_at_fixed_fpr_score
from models_under_pressure.probes.probes import ProbeFactory
from models_under_pressure.utils import double_check_config
from models_under_pressure.probes.sklearn_probes import SklearnProbe
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchDifferenceOfMeansClassifier,
)


def evaluate_probe_and_save_results(
    probe: Probe,
    train_dataset_path: Path,
    eval_dataset_name: str,
    eval_dataset: LabelledDataset,
    model_name: str,
    layer: int,
    output_dir: Path,
    save_results: bool = False,
    fpr: float = 0.01,
) -> tuple[list[float], DatasetResults]:
    """
    Evaluate a probe and save the results to a file.

    Args:
        probe: The probe to evaluate.
        train_dataset_path: The path to the train dataset.
        eval_dataset: The dataset to evaluate the probe on.
        layer: The layer to evaluate the probe on.
        output_dir: The directory to save the results to.
        save_results: Whether to save the results to a file.
        fpr: The FPR threshold to evaluate the probe at.
    Returns:
        A dictionary of the evaluated datasets and their results.

    Method designed to be used in the evaluate_probes.py experiment run
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    _, per_entry_probe_scores = probe.predict_proba(eval_dataset)
    print(f"Obtained {len(per_entry_probe_scores)} probe scores")

    if save_results:
        # We don't seem to use these fields for the main evaluation
        per_token_probe_scores = probe.per_token_predictions(
            inputs=eval_dataset.inputs,
        )

        # Get rid of the padding in the per token probe scores
        per_token_probe_scores = [
            probe_score[probe_score != -1] for probe_score in per_token_probe_scores
        ]

        # calculate logits for the per token probe scores
        per_token_probe_logits = [
            (np.log(probe_score) / (1 - probe_score + 1e-7)).tolist()
            for probe_score in per_token_probe_scores
        ]

        per_entry_probe_logits = [
            (
                np.log(per_entry_probe_score) / (1 - per_entry_probe_score + 1e-7)
            ).tolist()
            for per_entry_probe_score in per_entry_probe_scores
        ]

        # Assert no NaN values in the per token probe logits
        for i, logits in enumerate(per_token_probe_logits):
            if np.any(np.isnan(logits)):
                print(f"Found NaN values in probe logits for entry {i}")
            assert not np.any(np.isnan(logits)), "Found NaN values in probe logits"

        probe_scores_dict = {
            "per_entry_probe_scores": per_entry_probe_scores,
            "per_entry_probe_logits": per_entry_probe_logits,
            "per_token_probe_logits": per_token_probe_logits,
            "per_token_probe_scores": per_token_probe_scores,
        }

        for score, values in probe_scores_dict.items():
            if len(values) != len(eval_dataset.inputs):
                breakpoint()
            assert (
                len(values) == len(eval_dataset.inputs)
            ), f"{score} has length {len(values)} but eval_dataset has length {len(eval_dataset.inputs)}"

        try:
            dataset_with_probe_scores = LabelledDataset.load_from(
                output_dir / f"{eval_dataset_name}.jsonl"
            )
        except FileNotFoundError:
            dataset_with_probe_scores = eval_dataset.drop_cols(
                "activations", "input_ids", "attention_mask"
            )

        extra_fields = dict(**dataset_with_probe_scores.other_fields)

        short_model_name = model_name.split("/")[-1]
        column_name_template = f"_{short_model_name}_{train_dataset_path.stem}_l{layer}"

        for name, scores in probe_scores_dict.items():
            extra_fields[name + column_name_template] = scores

        dataset_with_probe_scores.other_fields = extra_fields

        # Save the dataset to the output path overriding the previous dataset
        print(
            f"Saving dataset to {EVALUATE_PROBES_DIR / f'{eval_dataset_name.split(".")[0]}.jsonl'}"
        )
        dataset_with_probe_scores.save_to(
            EVALUATE_PROBES_DIR / f"{eval_dataset_name.split('.')[0]}.jsonl",
            overwrite=True,
        )

    # Calculate the metrics for the dataset:
    auroc = roc_auc_score(
        eval_dataset.labels_numpy(),
        per_entry_probe_scores,
    )
    accuracy = accuracy_score(
        eval_dataset.labels_numpy(),
        np.array(per_entry_probe_scores) > 0.5,
    )

    tpr_at_fpr = tpr_at_fixed_fpr_score(
        eval_dataset.labels_numpy(),
        per_entry_probe_scores,
        fpr=fpr,
    )

    metrics = {
        "auroc": float(auroc),
        "accuracy": float(accuracy),
        "tpr_at_fpr": float(tpr_at_fpr),
        "fpr": float(fpr),
    }

    return per_entry_probe_scores, DatasetResults(layer=layer, metrics=metrics)


def get_coefs(probe: Probe) -> list[float]:
    if isinstance(probe, SklearnProbe):
        coefs = list(probe._classifier.named_steps["logisticregression"].coef_)  # type: ignore
    elif isinstance(probe, PytorchProbe):
        if isinstance(probe._classifier, PytorchDifferenceOfMeansClassifier):
            # For difference of means classifier, weights are directly in the linear layer
            coefs = list(probe._classifier.model.weight.data.cpu().numpy().flatten())  # type: ignore
        else:
            # For regular PyTorch probe, weights are in the second layer of Sequential
            coefs = list(probe._classifier.model[1].weight.data.cpu().numpy())  # type: ignore
    return coefs


def evaluate_probe_and_save_results(
    probe: Probe,
    train_dataset_path: Path,
    eval_dataset_name: str,
    eval_dataset: LabelledDataset,
    model_name: str,
    layer: int,
    output_dir: Path,
    save_results: bool = False,
    fpr: float = 0.01,
) -> tuple[list[float], DatasetResults]:
    """
    Evaluate a probe and save the results to a file.

    Args:
        probe: The probe to evaluate.
        train_dataset_path: The path to the train dataset.
        eval_dataset_name: The name of the dataset to evaluate the probe on.
        eval_dataset: The dataset to evaluate the probe on.
        model_name: The name of the model to evaluate the probe on.
        layer: The layer to evaluate the probe on.
        output_dir: The directory to save the results to.
        save_results: Whether to save the results to a file.
        fpr: The FPR threshold to evaluate the probe at.
    Returns:
        The per-entry probe scores and the results of the probe evaluation.

    Method designed to be used in the evaluate_probes.py experiment run
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    _, per_entry_probe_scores = probe.predict_proba(eval_dataset)
    print(f"Obtained {len(per_entry_probe_scores)} probe scores")

    if save_results:
        # We don't seem to use these fields for the main evaluation
        per_token_probe_scores = probe.per_token_predictions(
            inputs=eval_dataset.inputs,
        )

        # Get rid of the padding in the per token probe scores
        per_token_probe_scores = [
            probe_score[probe_score != -1] for probe_score in per_token_probe_scores
        ]

        # calculate logits for the per token probe scores
        per_token_probe_logits = [
            (np.log(probe_score) / (1 - probe_score + 1e-7)).tolist()
            for probe_score in per_token_probe_scores
        ]

        per_entry_probe_logits = [
            (
                np.log(per_entry_probe_score) / (1 - per_entry_probe_score + 1e-7)
            ).tolist()
            for per_entry_probe_score in per_entry_probe_scores
        ]

        # Assert no NaN values in the per token probe logits
        for i, logits in enumerate(per_token_probe_logits):
            if np.any(np.isnan(logits)):
                print(f"Found NaN values in probe logits for entry {i}")
            assert not np.any(np.isnan(logits)), "Found NaN values in probe logits"

        probe_scores_dict = {
            "per_entry_probe_scores": per_entry_probe_scores,
            "per_entry_probe_logits": per_entry_probe_logits,
            "per_token_probe_logits": per_token_probe_logits,
            "per_token_probe_scores": per_token_probe_scores,
        }

        for score, values in probe_scores_dict.items():
            if len(values) != len(eval_dataset.inputs):
                breakpoint()
            assert (
                len(values) == len(eval_dataset.inputs)
            ), f"{score} has length {len(values)} but eval_dataset has length {len(eval_dataset.inputs)}"

        try:
            dataset_with_probe_scores = LabelledDataset.load_from(
                output_dir / f"{eval_dataset_name}.jsonl"
            )
        except FileNotFoundError:
            dataset_with_probe_scores = eval_dataset.drop_cols(
                "activations", "input_ids", "attention_mask"
            )

        extra_fields = dict(**dataset_with_probe_scores.other_fields)

        short_model_name = model_name.split("/")[-1]
        column_name_template = f"_{short_model_name}_{train_dataset_path.stem}_l{layer}"

        for name, scores in probe_scores_dict.items():
            extra_fields[name + column_name_template] = scores

        dataset_with_probe_scores.other_fields = extra_fields

        # Save the dataset to the output path overriding the previous dataset
        print(
            f"Saving dataset to {EVALUATE_PROBES_DIR / f'{eval_dataset_name.split(".")[0]}.jsonl'}"
        )
        dataset_with_probe_scores.save_to(
            EVALUATE_PROBES_DIR / f"{eval_dataset_name.split('.')[0]}.jsonl",
            overwrite=True,
        )

    # Calculate the metrics for the dataset:
    auroc = roc_auc_score(
        eval_dataset.labels_numpy(),
        per_entry_probe_scores,
    )
    accuracy = accuracy_score(
        eval_dataset.labels_numpy(),
        np.array(per_entry_probe_scores) > 0.5,
    )

    tpr_at_fpr = tpr_at_fixed_fpr_score(
        eval_dataset.labels_numpy(),
        per_entry_probe_scores,
        fpr=fpr,
    )

    metrics = {
        "auroc": float(auroc),
        "accuracy": float(accuracy),
        "tpr_at_fpr": float(tpr_at_fpr),
        "fpr": float(fpr),
    }

    return per_entry_probe_scores, DatasetResults(layer=layer, metrics=metrics)


def run_evaluation(config: EvalRunConfig) -> list[EvaluationResult]:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    train_dataset, _ = load_train_test(
        dataset_path=config.dataset_path,
        n_per_class=config.max_samples,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )

    # Create the probe:
    print("Creating probe ...")
    probe = ProbeFactory.build(
        probe=config.probe_spec,
        train_dataset=train_dataset,
    )

    del train_dataset

    results_list = []

    for eval_dataset_path in tqdm(
        config.eval_datasets, desc="Evaluating on eval datasets"
    ):
        eval_dataset_name = eval_dataset_path.stem
        print(f"Loading eval dataset {eval_dataset_name} from {eval_dataset_path}")
        eval_dataset, _ = load_train_test(
            dataset_path=eval_dataset_path,
            model_name=config.model_name,
            layer=config.layer,
            compute_activations=config.compute_activations,
            n_per_class=config.max_samples,
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

    print(f"Saving results to {EVALUATE_PROBES_DIR / config.output_filename}")
    for result in results_list:
        result.save_to(EVALUATE_PROBES_DIR / config.output_filename)

    return results_list


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
        dataset_path=SYNTHETIC_DATASET_PATH,
        eval_datasets=list(EVAL_DATASETS.values()),
    )

    double_check_config(config)
    run_evaluation(config=config)

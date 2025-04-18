# Code to generate Figure 2
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from models_under_pressure.config import (
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
    EvalRunConfig,
)
from models_under_pressure.dataset_utils import load_dataset, load_splits_lazy
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import DatasetResults, EvaluationResult
from models_under_pressure.probes.base import Probe
from models_under_pressure.probes.metrics import tpr_at_fixed_fpr_score
from models_under_pressure.probes.probe_factory import ProbeFactory
from models_under_pressure.probes.pytorch_classifiers import (
    AttentionProbeAttnThenLinear,
    AttentionProbeAttnWeightLogits,
    PytorchAttentionClassifier,
    PytorchDifferenceOfMeansClassifier,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import SklearnProbe
from models_under_pressure.utils import double_check_config


def inv_softmax(x: list[np.ndarray]) -> list[list[float]]:
    return [np.log(x_i / (1 - x_i + 1e-7)).tolist() for x_i in x]


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

    per_entry_probe_scores = probe.predict_proba(eval_dataset)
    print(f"Obtained {len(per_entry_probe_scores)} probe scores")

    if save_results:
        # Get rid of the padding in the per token probe scores
        per_token_probe_scores = [
            probe_score[probe_score != -1]
            for probe_score in probe.per_token_predictions(eval_dataset.inputs)
        ]

        # calculate logits for the per token probe scores
        per_token_probe_logits = inv_softmax(per_token_probe_scores)
        per_entry_probe_logits = inv_softmax(per_entry_probe_scores)

        # Assert no NaN values in the per token probe logits
        for i, logits in enumerate(per_token_probe_logits):
            if np.any(np.isnan(logits)):
                raise ValueError(f"Found NaN values in probe logits for entry {i}")

        probe_scores_dict = {
            "per_entry_probe_scores": per_entry_probe_scores,
            "per_entry_probe_logits": per_entry_probe_logits,
            "per_token_probe_logits": per_token_probe_logits,
            "per_token_probe_scores": per_token_probe_scores,
        }

        for score, values in probe_scores_dict.items():
            if len(values) != len(eval_dataset.inputs):
                raise ValueError(
                    f"{score} has length {len(values)} "
                    f"but {eval_dataset_name} has length {len(eval_dataset.inputs)}"
                )

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

    y_true = eval_dataset.labels_numpy()
    y_pred = np.array(per_entry_probe_scores)

    # Calculate the metrics for the dataset:
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred > 0.5)),
        "tpr_at_fpr": float(tpr_at_fixed_fpr_score(y_true, y_pred, fpr=fpr)),
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
        elif isinstance(probe._classifier, PytorchAttentionClassifier):
            # For attention probe, get the weights from the final linear layer
            model = probe._classifier.model
            if isinstance(model, AttentionProbeAttnThenLinear):
                coefs = list(model.linear.weight.data.cpu().numpy().flatten())  # type: ignore
            elif isinstance(model, AttentionProbeAttnWeightLogits):
                coefs = list(model.linear.weight.data.cpu().numpy().flatten())  # type: ignore
            else:
                raise ValueError(f"Unknown attention probe model type: {type(model)}")
        else:
            # For regular PyTorch probe, weights are in the second layer of Sequential
            coefs = list(probe._classifier.model[1].weight.data.cpu().numpy())  # type: ignore
    return coefs


def run_evaluation(
    config: EvalRunConfig,
) -> tuple[list[EvaluationResult], list[float]]:
    """Train a linear probe on our training dataset and evaluate on all eval datasets."""
    splits = load_splits_lazy(
        dataset_path=config.dataset_path,
        dataset_filters=config.dataset_filters,
        n_per_class=config.max_samples,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )

    if isinstance(config.validation_dataset, Path):
        validation_dataset = load_dataset(
            dataset_path=config.validation_dataset,
            dataset_filters=config.dataset_filters,
            model_name=config.model_name,
            layer=config.layer,
            compute_activations=config.compute_activations,
            n_per_class=config.max_samples // 2 if config.max_samples else None,
        )
    elif config.validation_dataset:
        validation_dataset = splits["test"]
    else:
        validation_dataset = None

    # Create the probe:
    print("Creating probe ...")
    probe = ProbeFactory.build(
        probe_spec=config.probe_spec,
        train_dataset=splits["train"],
        validation_dataset=validation_dataset,
    )

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

        best_epoch = (
            probe._classifier.best_epoch
            if (
                isinstance(probe, PytorchProbe)
                and hasattr(probe._classifier, "best_epoch")
            )
            else None
        )
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
            best_epoch=best_epoch,
        )

        results_list.append(dataset_results)

        del eval_dataset

    coefs = get_coefs(probe)

    print(f"Saving results to {EVALUATE_PROBES_DIR / config.output_filename}")
    for result in results_list:
        result.save_to(EVALUATE_PROBES_DIR / config.output_filename)

    coefs_dict = {
        "id": config.id,
        "coefs": coefs[0].tolist(),  # type: ignore
    }
    with open(EVALUATE_PROBES_DIR / config.coefs_filename, "w") as f:
        json.dump(coefs_dict, f)

    return results_list, coefs


if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    np.random.seed(RANDOM_SEED)

    config = EvalRunConfig(
        layer=31,
        max_samples=None,
        model_name=LOCAL_MODELS["llama-70b"],
        probe_spec=ProbeSpec(
            name="pytorch_attention_probe",
            hyperparams={
                "batch_size": 16,
                "epochs": 50,
                "device": "cpu",
                "optimizer_args": {
                    "lr": 1e-3,
                    "weight_decay": 0.0004,
                },
                "attn_hidden_dim": 27,
                "probe_architecture": "attention_then_linear",
                "scheduler_decay": 0.62,
            },
        ),
        compute_activations=False,
        # dataset_path=TRAIN_DIR / "prompts_25_03_25_gpt-4o_original_plus_new.jsonl",
        dataset_path=SYNTHETIC_DATASET_PATH,
        # dataset_path=INPUTS_DIR / "combined_deployment_dataset.jsonl",
        # validation_dataset=SYNTHETIC_DATASET_PATH,
        validation_dataset=True,
        eval_datasets=list(EVAL_DATASETS.values()),
    )

    double_check_config(config)

    print(f"Running probe evaluation with ID {config.id}")
    print(f"Results will be saved to {EVALUATE_PROBES_DIR / config.output_filename}")
    run_evaluation(config=config)

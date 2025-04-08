import random
from pathlib import Path

import dotenv
import numpy as np
from jaxtyping import Float
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm

from models_under_pressure.config import EVALUATE_PROBES_DIR
from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import DatasetResults
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import (
    Probe,
    SklearnProbe,
)

# Set random seed for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

dotenv.load_dotenv()


def train_probes(
    model: LLMModel, dataset: LabelledDataset, layers: list[int] | None = None
) -> dict[int, Probe]:
    """Train a probe for each layer in the model.

    Args:
        model: The model to train the probe on.
        dataset: The dataset to train the probe on.
        layers: The layers to train the probe on.

    Returns:
        A dictionary of the trained probes.

    Used in the generate_heatmaps script...
    """

    layers = layers or list(range(model.n_layers))
    aggregator = Aggregator(
        preprocessor=Preprocessors.mean,
        postprocessor=Postprocessors.sigmoid,
    )

    if any(label == Label.AMBIGUOUS for label in dataset.labels):
        raise ValueError("Training dataset contains ambiguous labels")

    # Iterate over layers. For each layer, create a config, then train a probe and store it
    return {
        layer: SklearnProbe(aggregator=aggregator).fit(dataset)
        for layer in tqdm(layers, desc="Training probes")
    }


def tpr_at_fixed_fpr_score(
    y_true: Float[np.ndarray, " batch_size"],
    y_pred: Float[np.ndarray, " batch_size"],
    fpr: float,
) -> float:
    """Calculate TPR at a fixed FPR threshold.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        fpr: Target false positive rate threshold

    Returns:
        TPR value at the specified FPR threshold
    """
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_pred)

    # Find the TPR value at the closest FPR to our target
    idx = np.argmin(np.abs(fpr_vals - fpr))
    return float(tpr_vals[idx])


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
        coefs = list(probe._classifier.model.weight.data.cpu().numpy())  # type: ignore
    return coefs

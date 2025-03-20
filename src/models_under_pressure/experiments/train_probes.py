import random
from pathlib import Path

import dotenv
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from models_under_pressure.config import CACHE_DIR, MODEL_MAX_MEMORY
from models_under_pressure.experiments.dataset_splitting import (
    create_cross_validation_splits,
)
from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import Label, LabelledDataset
from models_under_pressure.interfaces.results import DatasetResults
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.sklearn_probes import (
    LinearProbe,
    compute_accuracy,
    load_or_train_probe,
)

# Set random seed for reproducibility
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

dotenv.load_dotenv()


def train_probes(
    model: LLMModel, dataset: LabelledDataset, layers: list[int] | None = None
) -> dict[int, LinearProbe]:
    """Train a probe for each layer in the model."""

    layers = layers or list(range(model.n_layers))
    aggregator = Aggregator(
        preprocessor=Preprocessors.mean,
        postprocessor=Postprocessors.sigmoid,
    )

    if any(label == Label.AMBIGUOUS for label in dataset.labels):
        raise ValueError("Training dataset contains ambiguous labels")

    # Iterate over layers. For each layer, create a config, then train a probe and store it
    return {
        layer: LinearProbe(_llm=model, layer=layer, aggregator=aggregator).fit(dataset)
        for layer in tqdm(layers, desc="Training probes")
    }


def cross_validate_probe(
    probe: LinearProbe, dataset_splits: list[LabelledDataset]
) -> np.ndarray:
    accuracies = []

    for dataset in dataset_splits:
        activations_obj = probe._llm.get_batched_activations(
            dataset=dataset,
            layer=probe.layer,
        )
        accuracy = compute_accuracy(
            probe,
            dataset,
            activations_obj,
        )
        accuracies.append(accuracy)

    return np.mean(np.array(accuracies), axis=1)


def cross_validate_probes(
    probes: list[LinearProbe], dataset: LabelledDataset
) -> np.ndarray:
    dataset_splits = create_cross_validation_splits(dataset)
    accuracies = np.array(
        [cross_validate_probe(probe, dataset_splits) for probe in probes]
    )
    return np.mean(accuracies, axis=1)


def train_probes_and_save_results(
    model_name: str,
    train_dataset: LabelledDataset,
    train_dataset_path: Path,
    eval_datasets: dict[str, LabelledDataset],
    aggregator: Aggregator,
    layer: int,
    output_dir: Path,
    save_results: bool = False,
) -> dict[str, tuple[LabelledDataset, DatasetResults]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = LLMModel.load(
        model_name,
        model_kwargs={
            "device_map": "auto",
            "max_memory": MODEL_MAX_MEMORY[model_name],
            "cache_dir": CACHE_DIR,
        },
    )
    probe = load_or_train_probe(
        model=model,
        train_dataset=train_dataset,
        train_dataset_path=train_dataset_path,
        layer=layer,
        aggregator=aggregator,
    )
    probe_scores_dict = {}

    for eval_dataset_name, eval_dataset in tqdm(
        eval_datasets.items(),
        desc=f"Evaluating datasets for layer {layer}",
        leave=False,
    ):
        per_entry_probe_scores = probe.predict_proba(eval_dataset)

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
                breakpoint()
            assert not np.any(np.isnan(logits)), "Found NaN values in probe logits"

        probe_scores_dict[eval_dataset_name] = {
            "per_entry_probe_logits": per_entry_probe_logits,
            "per_entry_probe_scores": per_entry_probe_scores,
            "per_token_probe_logits": per_token_probe_logits,
            "per_token_probe_scores": per_token_probe_scores,
        }

        for score, values in probe_scores_dict[eval_dataset_name].items():
            if len(values) != len(eval_dataset.inputs):
                breakpoint()
            assert (
                len(values) == len(eval_dataset.inputs)
            ), f"{score} has length {len(values)} but eval_dataset has length {len(eval_dataset.inputs)}"

    outputs = {}
    # Eval part of the function:
    for eval_dataset_name in eval_datasets.keys():
        if save_results:
            try:
                dataset_with_probe_scores = LabelledDataset.load_from(
                    output_dir / f"{eval_dataset_name}.jsonl"
                )
            except FileNotFoundError:
                dataset_with_probe_scores = eval_datasets[eval_dataset_name]
        else:
            dataset_with_probe_scores = eval_datasets[eval_dataset_name]

        extra_fields = dict(**dataset_with_probe_scores.other_fields)

        short_model_name = model.name.split("/")[-1]
        column_name_template = f"_{short_model_name}_{train_dataset_path.stem}_l{layer}"

        for name, scores in probe_scores_dict[eval_dataset_name].items():
            extra_fields[name + column_name_template] = scores

        dataset_with_probe_scores.other_fields = extra_fields

        # Save the dataset to the output path overriding the previous dataset
        print(
            f"Saving dataset to {output_dir / f'{eval_dataset_name.split(".")[0]}.jsonl'}"
        )
        dataset_with_probe_scores.save_to(
            output_dir / f"{eval_dataset_name.split('.')[0]}.jsonl", overwrite=True
        )

        # Calculate the metrics for the dataset:
        auroc = roc_auc_score(
            dataset_with_probe_scores.labels_numpy(),
            probe_scores_dict[eval_dataset_name]["per_entry_probe_scores"],
        )
        accuracy = accuracy_score(
            dataset_with_probe_scores.labels_numpy(),
            np.array(probe_scores_dict[eval_dataset_name]["per_entry_probe_scores"])
            > 0.5,
        )

        metrics = {
            "auroc": float(auroc),
            "accuracy": float(accuracy),
        }

        dataset_results = DatasetResults(layer=layer, metrics=metrics)

        outputs[eval_dataset_name] = (
            dataset_with_probe_scores,
            dataset_results,
        )

    return outputs


if __name__ == "__main__":
    pass
    # train_dataset = LabelledDataset.load_from(**GENERATED_DATASET)
    # eval_datasets = {
    #     name: LabelledDataset.load_from(path) for name, path in EVAL_DATASETS.items()
    # }
    # train_probes_and_save_results(
    #     model_name=LOCAL_MODELS["llama-8b"],
    #     train_dataset=train_dataset,
    #     train_dataset_path=GENERATED_DATASET["file_path_or_name"],
    #     eval_datasets=eval_datasets,
    #     layers=[7, 10, 12],
    #     output_dir=Path("data/results/train_probes"),
    # )

import json
from pathlib import Path

import pandas as pd


def process_raw_probe_results(result: dict) -> pd.DataFrame:
    """
    Process the raw probe results:
    1. Establish the size of the dataset, ensure that all fields match the dataset size.
    2. Hyperparameter fields are then duplicated for each element of the dataset.
    """

    list_fields = [
        "output_scores",
        "output_labels",
        "ground_truth_labels",
        "ids",
    ]
    data_size = len(result["output_scores"])

    for field in list_fields:
        assert (
            len(result.get(field, [])) == data_size
        ), f"Field {field} has a different size than the dataset. Expected {data_size}, got {len(result[field])}"

    # Config fields
    config_str = json.dumps(result["config"])
    probe_spec_str = json.dumps(result["config"].get(" probe_spec", {}))
    probe_name = result["config"].get("probe_spec", {}).get("name", "")

    if probe_name == "" and "method" in list(result.keys()):
        probe_name = result["method"]

    model_name = result["config"]["model_name"]
    max_samples = result["config"].get("max_samples", None)
    timestamp = result["timestamp"]

    if result.get("train_dataset_size", None) is not None:
        train_dataset_size = result["train_dataset_size"]
        output = {
            "config": [config_str] * data_size,
            "probe_spec": [probe_spec_str] * data_size,
            "probe_name": [probe_name] * data_size,
            "output_scores": result["output_scores"],
            "output_labels": result["output_labels"],
            "ground_truth_labels": result["ground_truth_labels"],
            "ids": result["ids"],
            "dataset_name": [result["dataset_name"]] * data_size,
            "dataset_path": [result["dataset_path"]] * data_size,
            "model_name": [model_name] * data_size,
            "max_samples": [max_samples] * data_size,
            "timestamp": [timestamp] * data_size,
            "train_dataset_size": [train_dataset_size] * data_size,
        }
    else:
        output = {
            "config": [config_str] * data_size,
            "probe_spec": [probe_spec_str] * data_size,
            "probe_name": [probe_name] * data_size,
            "output_scores": result["output_scores"],
            "output_labels": result["output_labels"],
            "ground_truth_labels": result["ground_truth_labels"],
            "ids": result["ids"],
            "dataset_name": [result["dataset_name"]] * data_size,
            "dataset_path": [result["dataset_path"]] * data_size,
            "model_name": [model_name] * data_size,
            "max_samples": [max_samples] * data_size,
            "timestamp": [timestamp] * data_size,
        }

    return pd.DataFrame(output)


def process_raw_continuation_results(result: dict) -> pd.DataFrame:
    """
    Process the raw continuation results:
    1. Establish the size of the dataset, ensure that all fields match the dataset size.
    2. Hyperparameter fields are then duplicated for each element of the dataset.
    """

    list_fields = [
        "low_stakes_scores",
        "high_stakes_scores",
        "labels",
        "ground_truth",
        "ids",
    ]
    data_size = len(result["low_stakes_scores"])

    for field in list_fields:
        assert (
            len(result.get(field, [])) == data_size
        ), f"Field {field} has a different size than the dataset. Expected {data_size}, got {len(result[field])}"

    dataset_name = result["dataset_name"]
    dataset_path = result["dataset_path"]
    model_name = result["model_name"]
    max_samples = result["max_samples"]
    timestamp = result["timestamp"]

    return pd.DataFrame(
        {
            "low_stakes_scores": result["low_stakes_scores"],
            "high_stakes_scores": result["high_stakes_scores"],
            "labels": result["labels"],
            "ground_truth": result["ground_truth"],
            "ids": result["ids"],
            "dataset_name": [dataset_name] * data_size,
            "dataset_path": [dataset_path] * data_size,
            "model_name": [model_name] * data_size,
            "max_samples": [max_samples] * data_size,
            "timestamp": [timestamp] * data_size,
        }
    )


def process_raw_finetune_results(result: dict) -> pd.DataFrame:
    """
    Process the raw finetune results:
    1. Establish the size of the dataset, ensure that all fields match the dataset size.
    2. Hyperparameter fields are then duplicated for each element of the dataset.
    """

    list_fields = [
        "labels",
        "ground_truth",
        "scores",
        "ids",
    ]
    data_size = len(result["scores"])

    for field in list_fields:
        assert (
            len(result.get(field, [])) == data_size
        ), f"Field {field} has a different size than the dataset. Expected {data_size}, got {len(result[field])}"

    dataset_name = result["dataset_name"]
    dataset_path = result["dataset_path"]
    model_name = result["model_name"]
    max_samples = result["max_samples"]
    timestamp = result["timestamp"]

    if result.get("dataset_size", None) is not None:
        train_dataset_size = result["dataset_size"]

        output = {
            "scores": result["scores"],
            "labels": result["labels"],
            "ground_truth": result["ground_truth"],
            "ids": result["ids"],
            "dataset_name": [dataset_name] * data_size,
            "dataset_path": [dataset_path] * data_size,
            "model_name": [model_name] * data_size,
            "max_samples": [max_samples] * data_size,
            "timestamp": [timestamp] * data_size,
            "train_dataset_size": [train_dataset_size] * data_size,
        }
    else:
        output = {
            "scores": result["scores"],
            "labels": result["labels"],
            "ground_truth": result["ground_truth"],
            "ids": result["ids"],
            "dataset_name": [dataset_name] * data_size,
            "dataset_path": [dataset_path] * data_size,
            "model_name": [model_name] * data_size,
            "max_samples": [max_samples] * data_size,
            "timestamp": [timestamp] * data_size,
        }

    return pd.DataFrame(output)


def get_probe_results(probe_result_paths: list[Path]) -> pd.DataFrame:
    """
    Read the results file, return a dataframe where each row is an entry in a dataset.

    'config', 'dataset_name', 'dataset_path', 'metrics', 'method',
    'best_epoch', 'output_scores', 'output_labels', 'ground_truth_labels',
    'ground_truth_scale_labels', 'token_counts', 'ids', 'mean_of_masked_activations',
    'masked_activations', 'timestamp']

    """
    print("Loading probe results ... ")
    data_frames = []

    for i, probe_result_path in enumerate(probe_result_paths):
        results = [json.loads(line) for line in open(probe_result_path)]

        file_results = []
        for result in results:
            df = process_raw_probe_results(result)
            file_results.append(df)

        file_results = pd.concat(file_results, ignore_index=False)
        file_results["load_id"] = i
        data_frames.append(file_results)

    output = pd.concat(data_frames, ignore_index=False)
    output["dataset_name"] = output["dataset_name"].apply(map_dataset_name)
    return output


def get_baseline_results(baseline_result_paths: list[Path]) -> pd.DataFrame:
    """
    Read the baseline results file, return a dataframe with the following keys:

    'ids', 'accuracy', 'labels', 'ground_truth', 'ground_truth_scale_labels',
    'dataset_name', 'dataset_path', 'model_name', 'max_samples',
    'timestamp', 'scores', 'token_counts'

    for each result stored in the specified path.

    Args:
        baseline_result_paths: A dictionary mapping dataset names to paths to the baseline results.

    Returns:
        A dataframe with the baseline results.
    """
    print("Loading baseline results ... ")
    data_frames = []
    for i, path in enumerate(baseline_result_paths):
        with open(path) as f:
            results = [json.loads(line) for line in f if line.strip()]

        file_results = []
        for result in results:
            df = process_raw_finetune_results(result)
            file_results.append(df)

        file_results = pd.concat(file_results, ignore_index=False)
        file_results["load_id"] = i
        data_frames.append(file_results)

    output = pd.concat(data_frames, ignore_index=False)
    output["dataset_name"] = output["dataset_name"].apply(map_dataset_name)
    return output


def map_dataset_name(dataset: str | Path) -> str:
    if isinstance(dataset, Path):
        dataset = dataset.stem
    if "manual" in dataset:
        return "Manual"
    elif "anthropic" in dataset:
        return "Anthropic"
    elif "toolace" in dataset:
        return "Toolace"
    elif "mts" in dataset:
        return "MTS"
    elif "mt" in dataset:
        return "MT"
    elif "mask" in dataset:
        return "Mask"
    elif "mental_health" in dataset:
        return "Mental Health"
    elif "aya" in dataset or "redteaming" in dataset:
        return "Aya Redteaming"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_continuation_results(continuation_result_paths: list[Path]) -> pd.DataFrame:
    """
    Read the continuation results file
    """

    print("Loading continuation results ... ")
    data_frames = []

    for i, probe_result_path in enumerate(continuation_result_paths):
        results = [json.loads(line) for line in open(probe_result_path)]

        file_results = []
        for result in results:
            df = process_raw_continuation_results(result)
            file_results.append(df)

        file_results = pd.concat(file_results, ignore_index=False)
        file_results["load_id"] = i
        data_frames.append(file_results)

    output = pd.concat(data_frames, ignore_index=False)
    output["dataset_name"] = output["dataset_name"].apply(map_dataset_name)
    return output

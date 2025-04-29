import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

from models_under_pressure.baselines.continuation import (
    evaluate_likelihood_continuation_baseline,
    likelihood_continuation_prompts,
)
from models_under_pressure.config import (
    CONFIG_DIR,
    DATA_DIR,
    EVAL_DATASETS,
    LOCAL_MODELS,
)
from models_under_pressure.dataset_utils import load_dataset
from models_under_pressure.experiments.evaluate_probes import (
    calculate_metrics,
)
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.interfaces.results import (
    DatasetResults,
    EvalRunConfig,
    EvaluationResult,
    LikelihoodBaselineResults,
)
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.probe_factory import ProbeFactory
from models_under_pressure.utils import AttrDict


def get_model_baseline_prompt(model_name: str) -> str:
    """Get the baseline prompt for a given model from its config file."""
    config_path = CONFIG_DIR / "model" / f"{model_name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["baseline_prompt"]


def _run_monitoring_cascade(
    model_names: List[str],
    probe_model_name: str | None,
    probe_layer: int | None,
    dataset_names: List[str],
    train_dataset_path: Path,
    probe_spec: ProbeSpec,
    max_samples: Optional[int] = None,
    batch_size: int = 4,
    output_dir: Optional[Path] = None,
    compute_activations: bool = True,
) -> tuple[List[LikelihoodBaselineResults], List[EvaluationResult], Path]:
    """Run a monitoring cascade experiment that evaluates both baselines and a probe.

    Args:
        model_names: List of model names to evaluate baselines for
        dataset_name: Name of the dataset to evaluate on
        train_dataset_name: Name of the dataset to train the probe on
        probe_spec: Probe specification from Hydra config
        max_samples: Maximum number of samples to evaluate on. If None, uses all samples.
        batch_size: Batch size for evaluation
        output_dir: Directory to save results to. If None, creates a date-stamped directory in DATA_DIR/results
        compute_activations: Whether to compute activations for the probe

    Returns:
        Tuple of (baseline_results, probe_results, output_dir)
    """
    if max_samples is not None:
        print(
            "Warning: In the current implementation, max_samples leads to different samples for different methods and the analysis part won't work!"
        )

    # Create output directory
    if output_dir is None:
        date = datetime.now().strftime("%Y%m%d")
        output_dir = DATA_DIR / "results" / f"monitoring_cascade_{date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    # Run baselines for each model
    baseline_results = []
    for model_name in model_names:
        print(f"\nRunning baseline for model: {model_name}")

        # Get model-specific baseline prompt
        baseline_prompt = get_model_baseline_prompt(model_name)
        print(f"Using baseline prompt: {baseline_prompt}")

        # Load model
        model = LLMModel.load(LOCAL_MODELS[model_name])

        try:
            for dataset_name in dataset_names:
                print(
                    f"\nRunning baseline for model: {model_name} on dataset: {dataset_name}"
                )

                # Get dataset paths
                eval_dataset_path = EVAL_DATASETS[dataset_name]

                # We don't need any activations in this case!
                eval_dataset = LabelledDataset.load_from(eval_dataset_path)
                if max_samples is not None and len(eval_dataset) > max_samples:
                    eval_dataset = subsample_balanced_subset(
                        eval_dataset, n_per_class=max_samples // 2
                    )
                assert eval_dataset.other_fields["scale_labels"] is not None

                results = evaluate_likelihood_continuation_baseline(
                    model=model,
                    prompt_config=likelihood_continuation_prompts[baseline_prompt],
                    dataset_name=dataset_name,
                    dataset=eval_dataset,
                    dataset_path=eval_dataset_path,
                    max_samples=None,  # Don't sample again, use the pre-sampled dataset
                    batch_size=batch_size,
                )
                baseline_results.append(results)
                # Save baseline results
                results.save_to(output_dir / f"baseline_{model_name}.jsonl")
        finally:
            # Ensure model is unloaded and GPU memory is freed
            del model
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if probe_model_name:
        assert probe_layer is not None

        # Train probe
        train_dataset = load_dataset(
            dataset_path=train_dataset_path,
            model_name=probe_model_name,
            layer=probe_layer,
            compute_activations=compute_activations,
            n_per_class=max_samples // 2 if max_samples else None,
        )
        print("\nTraining probe...")
        probe = ProbeFactory.build(
            probe=probe_spec,
            train_dataset=train_dataset,
        )

        # Evaluate probe
        probe_results = []
        for dataset_name in dataset_names:
            eval_dataset = load_dataset(
                dataset_path=eval_dataset_path,
                model_name=probe_model_name,
                layer=probe_layer,
                compute_activations=compute_activations,
                n_per_class=max_samples // 2 if max_samples else None,
            )
            assert eval_dataset.other_fields["scale_labels"] is not None

            print("\nEvaluating probe...")
            probe_scores = probe.predict_proba(eval_dataset)  # .tolist()
            dataset_results = DatasetResults(
                layer=probe_layer,
                metrics=calculate_metrics(
                    eval_dataset.labels_numpy(), probe_scores, fpr=0.01
                ),
            )

            # Get the token counts from the activations array
            # token_counts = eval_dataset.other_fields["activations"].shape[1]
            token_counts = (
                np.array(eval_dataset.other_fields["attention_mask"])
                .sum(axis=1)
                .tolist()
            )

            # Create EvaluationResult with all scores and labels
            probe_result = EvaluationResult(
                config=EvalRunConfig(
                    id="monitoring_cascade",
                    layer=probe_layer,
                    probe_spec=probe_spec,
                    max_samples=max_samples,
                    dataset_path=train_dataset_path,
                    eval_datasets=[eval_dataset_path],
                    model_name=probe_model_name,
                    compute_activations=compute_activations,
                ),
                dataset_name=dataset_name,
                dataset_path=eval_dataset_path,
                metrics=dataset_results,
                method="linear_probe",
                output_scores=probe_scores.tolist(),
                output_labels=[int(score > 0.5) for score in probe_scores],
                ground_truth_labels=eval_dataset.labels_numpy().tolist(),
                token_counts=token_counts,
                ids=list(eval_dataset.ids),
                ground_truth_scale_labels=list(
                    eval_dataset.other_fields["scale_labels"]
                ),  # type: ignore
            )

            # Save probe results
            probe_result.save_to(output_dir / "probe_results.jsonl")
            probe_results.append(probe_result)

    return baseline_results, probe_results, output_dir


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="experiments/monitoring_cascade",
    version_base=None,
)
def run_monitoring_cascade(cfg: DictConfig) -> None:
    """Hydra wrapper for the monitoring cascade experiment.

    Args:
        cfg: Hydra configuration object
    """
    print(cfg)

    # Get probe layer from model config
    if cfg.probe_model_name:
        model_config_path = CONFIG_DIR / "model" / f"{cfg.probe_model_name}.yaml"
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        if cfg.probe_layer is None:
            cfg.probe_layer = model_config.get("layer")
        cfg.probe_model_name = model_config.get("name", cfg.probe_model_name)

    cfg = AttrDict(
        OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)  # type: ignore
    )

    output_dir = (
        Path(cfg.output_dir)
        or DATA_DIR
        / "results"
        / f"monitoring_cascade_{datetime.now().strftime('%Y%m%d')}"
    )

    if cfg.compute_results:
        # Create probe specification from Hydra config
        probe_spec = ProbeSpec(
            name=cfg.probe.name,
            hyperparams=cfg.probe.hyperparams,
        )

        # Convert train_dataset_path to Path object and resolve relative to DATA_DIR
        train_dataset_path = DATA_DIR / cfg.train_dataset_path

        baseline_results, probe_results, output_dir = _run_monitoring_cascade(
            model_names=cfg.model_names,
            probe_model_name=cfg.probe_model_name,
            probe_layer=cfg.probe_layer,
            dataset_names=cfg.eval_datasets
            if cfg.eval_datasets
            else EVAL_DATASETS.keys(),
            train_dataset_path=train_dataset_path,
            probe_spec=probe_spec,
            max_samples=cfg.max_samples,
            batch_size=cfg.batch_size,
            compute_activations=cfg.compute_activations,
        )

    if cfg.analyze_cascade:
        results_file = output_dir / "cascade_results.jsonl"
        # Load existing results if needed
        baseline_results, probe_results = load_existing_results(output_dir)
        # Remove existing results file if it exists
        if results_file.exists():
            results_file.unlink()
        compute_cascade_results(
            baseline_results_by_dataset=baseline_results,
            probe_results_by_dataset=probe_results,
            results_file=results_file,
            first_baseline_model_name=cfg.first_baseline_model_name,
            target_dataset=cfg.target_dataset,
        )
        # Always generate plot after analyzing cascade results
        plot_cascade_results(
            results_file,
            output_file=output_dir / "cascade_plot.pdf",
            target_dataset=cfg.target_dataset,
            show_difference_from_probe=cfg.show_difference_from_probe,
        )


class CascadeResults(BaseModel):
    scores: list[float]
    labels: list[int]
    ground_truth_labels: list[int]
    ground_truth_scores: list[int]
    auroc: float
    flops: list[int]

    def __init__(
        self,
        scores: list[float],
        labels: list[int],
        ground_truth_labels: list[int],
        ground_truth_scores: list[int],
        flops: list[int],
    ):
        assert (
            len(scores)
            == len(labels)
            == len(ground_truth_labels)
            == len(ground_truth_scores)
            == len(flops)
        )

        super().__init__(
            scores=scores,
            labels=labels,
            ground_truth_labels=ground_truth_labels,
            ground_truth_scores=ground_truth_scores,
            auroc=float(roc_auc_score(ground_truth_labels, scores)),
            flops=flops,
        )

    def __add__(self, other: "CascadeResults") -> "CascadeResults":
        """Combine two CascadeResults objects using the + operator.

        Args:
            other: Another CascadeResults object to combine with

        Returns:
            A new CascadeResults object with combined data and recalculated AUROC
        """
        # Combine all lists
        combined_scores = self.scores + other.scores
        combined_labels = self.labels + other.labels
        combined_ground_truth_labels = (
            self.ground_truth_labels + other.ground_truth_labels
        )
        combined_ground_truth_scores = (
            self.ground_truth_scores + other.ground_truth_scores
        )
        combined_flops = self.flops + other.flops

        return CascadeResults(
            scores=combined_scores,
            labels=combined_labels,
            ground_truth_labels=combined_ground_truth_labels,
            ground_truth_scores=combined_ground_truth_scores,
            flops=combined_flops,
        )


def get_per_token_flops(full_model_name: str) -> int:
    model_name = next(k for k, v in LOCAL_MODELS.items() if v == full_model_name)
    # TODO These are from GPT-4.5 and should be verified
    if model_name == "llama-1b":
        per_token_flops = 1.3 * 10**9
    elif model_name == "llama-8b":
        per_token_flops = 9.6 * 10**9
    elif model_name == "llama-70b":
        per_token_flops = 85.9 * 10**9
    elif model_name == "gemma-1b":
        per_token_flops = 4.8 * 10**9
    elif model_name == "gemma-12b":
        per_token_flops = 10.9 * 10**9
    elif model_name == "gemma-27b":
        per_token_flops = 32.6 * 10**9
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return int(per_token_flops)


def evaluate_two_step_cascade(
    first_step_results: CascadeResults,  # Probe results
    second_step_results: CascadeResults,  # Baseline results
    fraction_of_samples: float = 0.2,
    merge_strategy: str = "baseline",
    selection_strategy: str = "top",
    remaining_strategy: str = "fixed_0",
    debug: bool = False,
) -> CascadeResults:
    """Combine two cascade results into a single cascade result.

    Args:
        first_step_results: Results from the first step (probe) of the cascade
        second_step_results: Results from the second step (baseline) of the cascade
        fraction_of_samples: Fraction of samples to use from first step
        merge_strategy: How to merge scores when both steps have results for the same sample
            Options: "baseline", "mean", "max"
        selection_strategy: How to select samples for first step
            Options: "top", "bottom", "mid"
        remaining_strategy: How to handle remaining samples
            Options: "fixed_X" where X is the fixed score, or "first"
        debug: Whether to print debug information
    """
    # Calculate number of samples to use
    total_samples = len(first_step_results.scores)
    num_samples_to_use = int(total_samples * fraction_of_samples)

    # Get indices based on selection strategy using probe scores
    sorted_indices = np.argsort(first_step_results.scores)
    if selection_strategy == "top":
        selected_indices = sorted_indices[-num_samples_to_use:]
    elif selection_strategy == "bottom":
        selected_indices = sorted_indices[:num_samples_to_use]
    elif selection_strategy == "mid":
        mid_point = len(sorted_indices) // 2
        half_samples = num_samples_to_use // 2
        start_idx = mid_point - half_samples
        end_idx = start_idx + num_samples_to_use
        selected_indices = sorted_indices[start_idx:end_idx]
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

    remaining_indices = np.setdiff1d(np.arange(total_samples), selected_indices)

    if debug:
        # Debug prints for token lengths and sample counts
        selected_flops = [second_step_results.flops[i] for i in selected_indices]
        avg_selected_flops = np.mean(selected_flops)
        print(f"\nStrategy: {selection_strategy}/{remaining_strategy}")
        print(f"Number of samples selected for baseline: {len(selected_indices)}")
        print(f"Average FLOPs of selected samples: {avg_selected_flops:.2f}")
        print(f"Total FLOPs in selected samples: {sum(selected_flops)}")

    # Process selected subset - use baseline results for selected samples
    if merge_strategy == "baseline":
        scores = [second_step_results.scores[i] for i in selected_indices]
        labels = [second_step_results.labels[i] for i in selected_indices]
    elif merge_strategy == "mean":
        scores = [
            (first_step_results.scores[i] + second_step_results.scores[i]) / 2
            for i in selected_indices
        ]
        labels = [score > 0.5 for score in scores]
    elif merge_strategy == "max":
        scores = [
            max(first_step_results.scores[i], second_step_results.scores[i])
            for i in selected_indices
        ]
        labels = [score > 0.5 for score in scores]
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    # For selected samples, FLOPs include both probe (for selection) and baseline
    selected_flops = [
        first_step_results.flops[i] + second_step_results.flops[i]
        for i in selected_indices
    ]

    top_results = CascadeResults(
        scores=scores,
        labels=labels,
        ground_truth_labels=[
            second_step_results.ground_truth_labels[i] for i in selected_indices
        ],
        ground_truth_scores=[
            second_step_results.ground_truth_scores[i] for i in selected_indices
        ],
        flops=selected_flops,
    )

    if len(remaining_indices) < 1:
        return top_results

    # Process remaining subset based on remaining strategy
    if remaining_strategy.startswith("fixed_"):
        fixed_score = float(remaining_strategy.split("_")[1])
        fixed_label = fixed_score > 0.5

        # For fixed strategy, we still need to account for probe FLOPs since it was used for selection
        remaining_flops = [first_step_results.flops[i] for i in remaining_indices]

        remaining_results = CascadeResults(
            scores=[fixed_score] * len(remaining_indices),
            labels=[fixed_label] * len(remaining_indices),
            ground_truth_labels=[
                first_step_results.ground_truth_labels[i] for i in remaining_indices
            ],
            ground_truth_scores=[
                first_step_results.ground_truth_scores[i] for i in remaining_indices
            ],
            flops=remaining_flops,
        )
    elif remaining_strategy == "first":
        # For first strategy, we use first step results and FLOPs
        remaining_results = CascadeResults(
            scores=[first_step_results.scores[i] for i in remaining_indices],
            labels=[first_step_results.labels[i] for i in remaining_indices],
            ground_truth_labels=[
                first_step_results.ground_truth_labels[i] for i in remaining_indices
            ],
            ground_truth_scores=[
                first_step_results.ground_truth_scores[i] for i in remaining_indices
            ],
            flops=[first_step_results.flops[i] for i in remaining_indices],
        )
    else:
        raise ValueError(f"Unknown remaining strategy: {remaining_strategy}")

    # Combine results
    return top_results + remaining_results


def evaluate_probe_baseline_cascade(
    baseline_results: LikelihoodBaselineResults,
    probe_results: EvaluationResult,
    fraction_of_samples: float = 0.2,
    merge_strategy: str = "baseline",
    selection_strategy: str = "top",
    remaining_strategy: str = "fixed_0",
    debug: bool = False,
) -> CascadeResults:
    assert probe_results.output_scores is not None
    assert probe_results.ground_truth_labels is not None
    assert probe_results.ground_truth_scale_labels is not None
    assert probe_results.token_counts is not None
    assert baseline_results.token_counts is not None

    assert probe_results.ids == baseline_results.ids

    # Get model-specific per_token_flops
    per_token_flops = get_per_token_flops(baseline_results.model_name)
    activation_dim = get_activation_dim(probe_results.config.model_name)

    # Create probe cascade results for all samples (first step)
    probe_flops = [activation_dim * count for count in probe_results.token_counts]
    probe_cascade = CascadeResults(
        scores=probe_results.output_scores,
        labels=[int(score > 0.5) for score in probe_results.output_scores],
        ground_truth_labels=probe_results.ground_truth_labels,
        ground_truth_scores=probe_results.ground_truth_scale_labels,
        flops=probe_flops,
    )

    # Create baseline cascade results for all samples (second step)
    baseline_flops = [
        int(per_token_flops) * count for count in baseline_results.token_counts
    ]
    baseline_cascade = CascadeResults(
        scores=baseline_results.high_stakes_scores,
        labels=baseline_results.labels,
        ground_truth_labels=baseline_results.ground_truth,
        ground_truth_scores=baseline_results.ground_truth_scale_labels,
        flops=baseline_flops,
    )

    # Combine the two cascade results
    return evaluate_two_step_cascade(
        first_step_results=probe_cascade,  # Probe is first step
        second_step_results=baseline_cascade,  # Baseline is second step
        fraction_of_samples=fraction_of_samples,
        merge_strategy=merge_strategy,
        selection_strategy=selection_strategy,
        remaining_strategy=remaining_strategy,
        debug=debug,
    )


def evaluate_single_baseline_cascade(
    baseline_results: LikelihoodBaselineResults,
    fraction_of_samples: float = 1.0,
) -> CascadeResults:
    assert baseline_results.ground_truth_scale_labels is not None
    assert baseline_results.ground_truth is not None
    assert baseline_results.token_counts is not None

    # Calculate number of samples to use
    total_samples = len(baseline_results.labels)
    num_samples_to_use = int(total_samples * fraction_of_samples)

    # Randomly sample indices for the subset
    np.random.seed(42)  # For reproducibility
    sampled_indices = np.random.choice(total_samples, num_samples_to_use, replace=False)
    remaining_indices = np.setdiff1d(np.arange(total_samples), sampled_indices)

    # Get model-specific per_token_flops
    per_token_flops = get_per_token_flops(baseline_results.model_name)

    # Process sampled subset
    sampled_flops = [
        int(per_token_flops) * baseline_results.token_counts[i] for i in sampled_indices
    ]
    sampled_results = CascadeResults(
        scores=[baseline_results.high_stakes_scores[i] for i in sampled_indices],
        labels=[baseline_results.labels[i] for i in sampled_indices],
        ground_truth_labels=[baseline_results.ground_truth[i] for i in sampled_indices],
        ground_truth_scores=[
            baseline_results.ground_truth_scale_labels[i] for i in sampled_indices
        ],
        flops=sampled_flops,
    )

    if len(remaining_indices) < 1:
        return sampled_results

    # Process remaining subset with fixed cascade
    remaining_results = evaluate_fixed_cascade(
        ground_truth_labels=[
            baseline_results.ground_truth[i] for i in remaining_indices
        ],
        ground_truth_scale_labels=[
            baseline_results.ground_truth_scale_labels[i] for i in remaining_indices
        ],
    )

    # Combine results
    return sampled_results + remaining_results


def get_activation_dim(full_model_name: str) -> int:
    model_name = next(k for k, v in LOCAL_MODELS.items() if v == full_model_name)
    if model_name == "llama-1b":
        activation_dim = 2048
    elif model_name == "gemma-1b":
        activation_dim = 2048
    elif model_name == "llama-8b":
        activation_dim = 4096
    elif model_name == "llama-70b":
        activation_dim = 8192
    elif model_name == "gemma-12b":
        activation_dim = 4096
    elif model_name == "gemma-27b":
        activation_dim = 4096
    else:
        raise ValueError(f"Unknown activation dimension for model: {model_name}")
    return int(activation_dim)


def evaluate_single_probe_cascade(
    evaluation_results: EvaluationResult,
) -> CascadeResults:
    assert evaluation_results.output_scores is not None
    assert evaluation_results.ground_truth_labels is not None
    assert evaluation_results.ground_truth_scale_labels is not None
    assert evaluation_results.token_counts is not None

    # Get activation dimension for model
    activation_dim = get_activation_dim(evaluation_results.config.model_name)

    # Depending on the probe, flops are calculated differently
    if evaluation_results.config.probe_spec.name in [
        "pytorch_per_entry_probe_mean",
        "sklearn_mean_agg_probe",
    ]:
        # Flops per token are approximately activation_dim
        # (Since the average has to be computed as well)
        flops = [
            activation_dim * token_count
            for token_count in evaluation_results.token_counts
        ]
    elif evaluation_results.config.probe_spec.name == "pytorch_per_token_probe":
        flops = [
            activation_dim * token_count
            for token_count in evaluation_results.token_counts
        ]
    else:
        raise ValueError(
            f"Unknown flops for probe type: {evaluation_results.config.probe_spec.name}"
        )

    results = CascadeResults(
        scores=evaluation_results.output_scores,  # type: ignore
        labels=evaluation_results.output_labels,  # type: ignore
        ground_truth_labels=evaluation_results.ground_truth_labels,  # type: ignore
        ground_truth_scores=evaluation_results.ground_truth_scale_labels,
        flops=flops,
    )
    return results


def evaluate_fixed_cascade(
    ground_truth_labels: list[int],
    ground_truth_scale_labels: list[int],
    fixed_score: float = 0.5,
    fixed_label: int = 0,
    flops: list[int] | None = None,
) -> CascadeResults:
    num_samples = len(ground_truth_labels)
    labels = [fixed_label] * num_samples
    scores = [fixed_score] * num_samples

    return CascadeResults(
        scores=scores,
        labels=labels,
        ground_truth_labels=ground_truth_labels,
        ground_truth_scores=ground_truth_scale_labels,
        flops=flops or [0] * num_samples,  # No computation
    )


def write_cascade_results_to_file(
    results: CascadeResults,
    output_file: Path,
    cascade_type: str,
    fraction_of_samples: float,
    model_name: Optional[str] = None,
    probe_model_name: Optional[str] = None,
    selection_strategy: Optional[str] = None,
    remaining_strategy: Optional[str] = None,
    merge_strategy: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> None:
    """Write cascade results to a JSONL file.

    Args:
        results: CascadeResults object containing the results
        output_file: Path to the output JSONL file
        cascade_type: Type of cascade (e.g., "baseline", "probe", "probe_baseline")
        model_name: Name of the baseline model used
        probe_model_name: Name of the probe model used (if applicable)
        fraction_of_samples: Fraction of samples used
        selection_strategy: Strategy used for selecting samples (if applicable)
        remaining_strategy: Strategy used for remaining samples (if applicable)
        dataset_name: Name of the dataset (if applicable)
    """
    # Compute average FLOPs per sample
    avg_flops_per_sample = sum(results.flops) / len(results.flops)

    result_dict = {
        "cascade_type": cascade_type,
        "baseline_model_name": model_name,
        "probe_model_name": probe_model_name,
        "fraction_of_samples": fraction_of_samples,
        "auroc": results.auroc,
        "avg_flops_per_sample": avg_flops_per_sample,
        "selection_strategy": selection_strategy,
        "remaining_strategy": remaining_strategy,
        "merge_strategy": merge_strategy,
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
    }

    with open(output_file, "a") as f:
        f.write(json.dumps(result_dict) + "\n")


def compute_cascade_results(
    baseline_results_by_dataset: dict[str, List[LikelihoodBaselineResults]],
    probe_results_by_dataset: dict[str, EvaluationResult],
    results_file: Path,
    first_baseline_model_name: Optional[str] = None,
    target_dataset: Optional[str] = None,
):
    """Compute cascade results for all datasets or a specific dataset.

    Args:
        baseline_results_by_dataset: Dictionary mapping dataset names to baseline results
        probe_results_by_dataset: Dictionary mapping dataset names to probe results
        results_file: Path to save results to
        first_baseline_model_name: Name of the first baseline model for two-step cascades
        target_dataset: If specified, only compute results for this dataset
    """
    fraction_of_sample_options = [0.1 * i for i in range(1, 11)]

    # Get datasets to process
    datasets = (
        [target_dataset] if target_dataset else baseline_results_by_dataset.keys()
    )

    for dataset_name in datasets:
        baseline_results = baseline_results_by_dataset.get(dataset_name, [])
        probe_results = probe_results_by_dataset.get(dataset_name)

        if not baseline_results:
            print(f"No baseline results found for dataset {dataset_name}")
            continue

        # Evaluate baseline cascades
        print(f"\nBaseline Results for {dataset_name}:")
        for result in baseline_results:
            for fraction_of_samples in fraction_of_sample_options + (
                [1.0] if 1.0 not in fraction_of_sample_options else []
            ):
                print(f"Model: {result.model_name}, Fraction: {fraction_of_samples}")
                cascade_results = evaluate_single_baseline_cascade(
                    result, fraction_of_samples=fraction_of_samples
                )
                print(f"- AUROC: {cascade_results.auroc:.3f}")
                print(f"- Total FLOPs: {sum(cascade_results.flops)}")

                # Write results to file
                write_cascade_results_to_file(
                    results=cascade_results,
                    output_file=results_file,
                    cascade_type="baseline",
                    model_name=result.model_name,
                    probe_model_name=None,
                    fraction_of_samples=fraction_of_samples,
                    dataset_name=dataset_name,
                )

        if probe_results:
            # Evaluate probe cascade
            print(f"\nProbe Results for {dataset_name}:")
            cascade_results = evaluate_single_probe_cascade(probe_results)
            print(f"- AUROC: {cascade_results.auroc:.3f}")
            print(f"- Total FLOPs: {sum(cascade_results.flops)}")

            # Write probe results to file
            write_cascade_results_to_file(
                results=cascade_results,
                output_file=results_file,
                cascade_type="probe",
                model_name=None,
                probe_model_name=probe_results.config.model_name,
                fraction_of_samples=1.0,
                dataset_name=dataset_name,
            )

            # Evaluate probe baseline cascade for all models
            selection_strategies = ["mid"]  # , "top", "bottom"]
            merge_strategies = ["mean"]  # , "max", "baseline"]
            remaining_strategies = ["first"]
            strategies = []
            for selection_strategy in selection_strategies:
                for merge_strategy in merge_strategies:
                    for remaining_strategy in remaining_strategies:
                        strategies.append(
                            {
                                "selection_strategy": selection_strategy,
                                "remaining_strategy": remaining_strategy,
                                "merge_strategy": merge_strategy,
                            }
                        )

            print(f"\nProbe+Baseline Cascade Results for {dataset_name}:")
            for baseline_result in baseline_results:
                for fraction_of_samples in fraction_of_sample_options:
                    for strategy in strategies:
                        print(
                            f"\nCascade with fraction of samples: {fraction_of_samples} (baseline model: {baseline_result.model_name})"
                        )
                        print(f"Strategy: {strategy}")
                        probe_baseline_cascade_results = (
                            evaluate_probe_baseline_cascade(
                                baseline_results=baseline_result,
                                probe_results=probe_results,
                                fraction_of_samples=fraction_of_samples,
                                **strategy,
                            )
                        )
                        print(f"- AUROC: {probe_baseline_cascade_results.auroc:.3f}")
                        print(
                            f"- Total FLOPs: {sum(probe_baseline_cascade_results.flops)}"
                        )

                        # Write probe baseline cascade results to file
                        write_cascade_results_to_file(
                            results=probe_baseline_cascade_results,
                            output_file=results_file,
                            cascade_type="probe_baseline",
                            model_name=baseline_result.model_name,
                            probe_model_name=probe_results.config.model_name,
                            fraction_of_samples=fraction_of_samples,
                            dataset_name=dataset_name,
                            **strategy,
                        )

        # Evaluate two-step baseline cascades if first_baseline_model_name is provided
        if first_baseline_model_name:
            print(f"\nTwo-Step Baseline Cascade Results for {dataset_name}:")
            # Find the first baseline results
            first_baseline_result = next(
                (
                    r
                    for r in baseline_results
                    if get_abbreviated_model_name(r.model_name)
                    == first_baseline_model_name
                ),
                None,
            )
            if first_baseline_result:
                # For each other baseline, create a two-step cascade
                for second_baseline_result in baseline_results:
                    if (
                        get_abbreviated_model_name(second_baseline_result.model_name)
                        != first_baseline_model_name
                    ):
                        for fraction_of_samples in fraction_of_sample_options:
                            strategy = {
                                "selection_strategy": "mid",
                                "remaining_strategy": "first",
                                "merge_strategy": "mean",
                            }
                            print(
                                f"\nTwo-step cascade with fraction of samples: {fraction_of_samples}"
                            )
                            print(
                                f"First baseline: {first_baseline_model_name}, Second baseline: {second_baseline_result.model_name}"
                            )
                            two_step_cascade_results = evaluate_two_baselines_cascade(
                                first_baseline_results=first_baseline_result,
                                second_baseline_results=second_baseline_result,
                                fraction_of_samples=fraction_of_samples,
                                **strategy,
                            )
                            print(f"- AUROC: {two_step_cascade_results.auroc:.3f}")
                            print(
                                f"- Total FLOPs: {sum(two_step_cascade_results.flops)}"
                            )

                            # Write two-step baseline cascade results to file
                            write_cascade_results_to_file(
                                results=two_step_cascade_results,
                                output_file=results_file,
                                cascade_type="two_step_baseline",
                                model_name=second_baseline_result.model_name,  # Use second baseline for color matching
                                probe_model_name=first_baseline_model_name,  # Use first baseline as probe_model_name
                                fraction_of_samples=fraction_of_samples,
                                dataset_name=dataset_name,
                                **strategy,
                            )


def get_abbreviated_model_name(full_name: str) -> str:
    """Convert full model name to abbreviated form using LOCAL_MODELS mapping."""
    if not full_name:
        return ""
    # Get the short name from LOCAL_MODELS
    model_name = next((k for k, v in LOCAL_MODELS.items() if v == full_name), full_name)
    return model_name


def plot_cascade_results(
    results_file: Path,
    output_file: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
    show_fraction_labels: bool = False,
    target_dataset: Optional[str] = None,
    show_difference_from_probe: bool = False,
) -> None:
    """Plot cascade results showing the tradeoff between FLOPs and AUROC.

    Args:
        results_file: Path to the JSONL file containing cascade results
        output_file: Path to save the plot. If None, saves to results_file.parent / "cascade_plot.pdf"
        figsize: Figure size as (width, height) in inches
        show_fraction_labels: Whether to show the fraction of samples labels on the plot points
        target_dataset: If specified, only plot results for this dataset
        show_difference_from_probe: If True, shows AUROC difference from probe performance. If False, shows absolute AUROC.
    """
    import json
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    # Read results from file
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                if (
                    target_dataset is None
                    or result.get("dataset_name") == target_dataset
                ):
                    results.append(result)

    # Group results by method and fraction
    grouped_results = defaultdict(lambda: defaultdict(list))
    probe_aurocs = defaultdict(list)  # Store probe AUROCs by dataset

    # First pass: collect probe AUROCs for each dataset
    for result in results:
        if result["cascade_type"] == "probe":
            dataset = result.get("dataset_name", "default")
            probe_aurocs[dataset].append(result["auroc"])

    # Calculate mean probe AUROC for each dataset
    mean_probe_aurocs = {
        dataset: float(np.mean(aurocs)) for dataset, aurocs in probe_aurocs.items()
    }

    # Second pass: process cascade results
    for result in results:
        if result["cascade_type"] == "probe":
            continue

        baseline_model = result.get("baseline_model_name")
        if baseline_model:
            # Create a unique key for each method
            if result["cascade_type"] in ["probe_baseline", "two_step_baseline"]:
                key = (
                    baseline_model,
                    result["cascade_type"],
                    result.get("selection_strategy", "top"),
                    result.get("remaining_strategy", "fixed_0"),
                    result.get("merge_strategy", "baseline"),
                )
            else:  # baseline
                key = (baseline_model, result["cascade_type"])

            # Group by fraction_of_samples
            fraction = result["fraction_of_samples"]
            dataset = result.get("dataset_name", "default")
            probe_auroc = mean_probe_aurocs.get(dataset, 0.0)

            # Calculate AUROC value based on show_difference_from_probe flag
            auroc_value = (
                result["auroc"] - probe_auroc
                if show_difference_from_probe
                else result["auroc"]
            )

            grouped_results[key][fraction].append(
                {
                    "auroc": auroc_value,
                    "avg_flops_per_sample": result["avg_flops_per_sample"],
                }
            )

    # Set up the plot
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    # Create color palette based on number of unique baseline models
    baseline_models = sorted(set(key[0] for key in grouped_results.keys()))
    colors = sns.color_palette("husl", n_colors=len(baseline_models))
    model_to_color = {model: color for model, color in zip(baseline_models, colors)}

    # Define line styles
    line_styles = {
        "baseline": "-",
        "probe_baseline": {
            "top": "--",
            "bottom": ":",
            "mid": "-.",
        },
        "two_step_baseline": {
            "top": "--",
            "bottom": ":",
            "mid": "-.",
        },
    }

    # Plot each group
    for key, fraction_results in grouped_results.items():
        baseline_model = key[0]
        cascade_type = key[1]
        color = model_to_color[baseline_model]

        # Get appropriate line style
        if cascade_type == "baseline":
            linestyle = line_styles["baseline"]
        else:  # probe_baseline or two_step_baseline
            selection_strategy = key[2]
            linestyle = line_styles[cascade_type][selection_strategy]

        # Sort fractions
        fractions = sorted(fraction_results.keys())

        # Compute means and standard deviations for each fraction
        mean_aurocs = []
        std_aurocs = []
        mean_flops_per_sample = []
        std_flops_per_sample = []

        for fraction in fractions:
            results = fraction_results[fraction]
            aurocs = [r["auroc"] for r in results]
            flops = [r["avg_flops_per_sample"] for r in results]

            mean_aurocs.append(float(np.mean(aurocs)))
            std_aurocs.append(float(np.std(aurocs)))
            mean_flops_per_sample.append(float(np.mean(flops)))
            std_flops_per_sample.append(float(np.std(flops)))

        # Create label
        if cascade_type == "baseline":
            label = f"Baseline ({get_abbreviated_model_name(baseline_model)})"
        elif cascade_type == "probe_baseline":
            selection_strategy = key[2]
            remaining_strategy = key[3]
            merge_strategy = key[4]
            remaining_strategy_display = (
                remaining_strategy.replace("fixed_", "fixed=")
                if remaining_strategy
                else "fixed_0"
            )
            label = f"Probe+Baseline ({get_abbreviated_model_name(baseline_model)}) - {selection_strategy}/{remaining_strategy_display}/{merge_strategy}"
        else:  # two_step_baseline
            selection_strategy = key[2]
            remaining_strategy = key[3]
            merge_strategy = key[4]
            first_baseline = key[4]  # probe_model_name is stored here
            remaining_strategy_display = (
                remaining_strategy.replace("fixed_", "fixed=")
                if remaining_strategy
                else "fixed_0"
            )
            label = f"Two-Step Baseline ({get_abbreviated_model_name(first_baseline)}→{get_abbreviated_model_name(baseline_model)}) - {selection_strategy}/{remaining_strategy_display}/{merge_strategy}"

        # Plot line with shaded region
        plt.plot(
            mean_flops_per_sample,
            mean_aurocs,
            "o-",
            color=color,
            linestyle=linestyle,
            label=label,
            markersize=3,
        )

        # Add shaded region for uncertainty
        plt.fill_between(
            mean_flops_per_sample,
            np.array(mean_aurocs) - np.array(std_aurocs),
            np.array(mean_aurocs) + np.array(std_aurocs),
            color=color,
            alpha=0.1,
        )

        # Add fraction labels if enabled
        if show_fraction_labels:
            for x, y, f in zip(mean_flops_per_sample, mean_aurocs, fractions):
                plt.annotate(
                    f"{f:.2f}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    # Plot probe performance line after all other data
    if show_difference_from_probe:
        # For difference plot, show zero line as probe performance
        plt.axhline(
            y=0, color="gray", linestyle="--", label="Probe Performance", alpha=0.5
        )
    elif probe_aurocs:
        # For absolute plot, show actual probe performance
        mean_probe_auroc = float(
            np.mean([auroc for aurocs in probe_aurocs.values() for auroc in aurocs])
        )
        std_probe_auroc = float(
            np.std([auroc for aurocs in probe_aurocs.values() for auroc in aurocs])
        )

        # Get the x-axis limits after plotting all the data
        x_min, x_max = plt.xlim()

        plt.axhline(
            y=mean_probe_auroc,
            color="gray",
            linestyle="--",
            label="Probe Performance",
            alpha=0.5,
        )

        plt.fill_between(
            [x_min, x_max],  # Use actual x-axis range
            mean_probe_auroc - std_probe_auroc,
            mean_probe_auroc + std_probe_auroc,
            color="gray",
            alpha=0.1,
        )

    # Customize plot
    plt.xlabel("Average FLOPs per Sample (log scale)", fontsize=12)
    plt.ylabel(
        "AUROC Difference from Probe" if show_difference_from_probe else "AUROC",
        fontsize=12,
    )
    title = "Cascade Performance Tradeoff"
    if target_dataset:
        title += f" - {target_dataset}"
    else:
        title += " (Averaged across datasets)"
    plt.title(title, fontsize=14, pad=20)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Set x-axis to log scale
    plt.xscale("log")

    # Format x-axis ticks to be more readable
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"10^{int(np.log10(x))}"))

    # Adjust layout
    plt.tight_layout()

    # Save plot
    if output_file is None:
        output_file = results_file.parent / "cascade_plot.pdf"
        if target_dataset:
            output_file = results_file.parent / f"cascade_plot_{target_dataset}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def load_existing_results(
    output_dir: Path,
) -> tuple[dict[str, List[LikelihoodBaselineResults]], dict[str, EvaluationResult]]:
    """Load existing results from a previous run.

    Args:
        output_dir: Directory containing the results files

    Returns:
        Tuple of (baseline_results_by_dataset, probe_results_by_dataset)
    """
    baseline_results_by_dataset = defaultdict(list)
    probe_results_by_dataset = {}

    # Read all baseline files
    for baseline_file in output_dir.glob("baseline_*.jsonl"):
        with open(baseline_file) as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    result = LikelihoodBaselineResults.model_validate_json(line.strip())
                    baseline_results_by_dataset[result.dataset_name].append(result)

    # Read probe results
    with open(output_dir / "probe_results.jsonl") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                result = EvaluationResult.model_validate_json(line.strip())
                probe_results_by_dataset[result.dataset_name] = result

    return baseline_results_by_dataset, probe_results_by_dataset


def evaluate_two_baselines_cascade(
    first_baseline_results: LikelihoodBaselineResults,
    second_baseline_results: LikelihoodBaselineResults,
    fraction_of_samples: float = 0.2,
    merge_strategy: str = "baseline",
    selection_strategy: str = "top",
    remaining_strategy: str = "fixed_0",
) -> CascadeResults:
    """Evaluate a cascade that uses two different baselines.

    The first baseline is used to select samples, and the second baseline is applied to those samples.
    The remaining samples are handled according to the remaining strategy.

    Args:
        first_baseline_results: Results from the first baseline (used for selection)
        second_baseline_results: Results from the second baseline (applied to selected samples)
        fraction_of_samples: Fraction of samples to use from first baseline
        merge_strategy: How to merge scores when both baselines have results for the same sample
            Options: "baseline", "mean", "max"
        selection_strategy: How to select samples for first baseline
            Options: "top", "bottom", "mid"
        remaining_strategy: How to handle remaining samples
            Options: "fixed_X" where X is the fixed score, or "baseline"
    """
    assert first_baseline_results.token_counts is not None
    assert second_baseline_results.token_counts is not None
    assert first_baseline_results.ids == second_baseline_results.ids

    # Get model-specific per_token_flops
    first_per_token_flops = get_per_token_flops(first_baseline_results.model_name)
    second_per_token_flops = get_per_token_flops(second_baseline_results.model_name)

    # Create first baseline cascade results for all samples (first step)
    first_baseline_flops = [
        int(first_per_token_flops) * count
        for count in first_baseline_results.token_counts
    ]
    first_baseline_cascade = CascadeResults(
        scores=first_baseline_results.high_stakes_scores,
        labels=first_baseline_results.labels,
        ground_truth_labels=first_baseline_results.ground_truth,
        ground_truth_scores=first_baseline_results.ground_truth_scale_labels,
        flops=first_baseline_flops,
    )

    # Create second baseline cascade results for all samples (second step)
    second_baseline_flops = [
        int(second_per_token_flops) * count
        for count in second_baseline_results.token_counts
    ]
    second_baseline_cascade = CascadeResults(
        scores=second_baseline_results.high_stakes_scores,
        labels=second_baseline_results.labels,
        ground_truth_labels=second_baseline_results.ground_truth,
        ground_truth_scores=second_baseline_results.ground_truth_scale_labels,
        flops=second_baseline_flops,
    )

    # Combine the two cascade results
    return evaluate_two_step_cascade(
        first_step_results=first_baseline_cascade,  # First baseline is first step
        second_step_results=second_baseline_cascade,  # Second baseline is second step
        fraction_of_samples=fraction_of_samples,
        merge_strategy=merge_strategy,
        selection_strategy=selection_strategy,
        remaining_strategy=remaining_strategy,
    )


if __name__ == "__main__":
    run_monitoring_cascade()

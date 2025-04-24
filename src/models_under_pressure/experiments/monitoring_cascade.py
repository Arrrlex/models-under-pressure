from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
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
    SYNTHETIC_DATASET_PATH,
    global_settings,
)
from models_under_pressure.dataset_utils import load_dataset
from models_under_pressure.experiments.evaluate_probes import (
    calculate_metrics,
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


def get_model_baseline_prompt(model_name: str) -> str:
    """Get the baseline prompt for a given model from its config file."""
    config_path = CONFIG_DIR / "model" / f"{model_name}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["baseline_prompt"]


def run_monitoring_cascade(
    model_names: List[str],
    probe_model_name: str,
    probe_layer: int,
    dataset_name: str,
    train_dataset_path: Path,
    probe_type: str = "pytorch_per_entry_probe_mean",
    max_samples: Optional[int] = None,
    batch_size: int = 4,
    output_dir: Optional[Path] = None,
    compute_activations: bool = True,
) -> tuple[List[LikelihoodBaselineResults], EvaluationResult]:
    """Run a monitoring cascade experiment that evaluates both baselines and a probe.

    Args:
        model_names: List of model names to evaluate baselines for
        dataset_name: Name of the dataset to evaluate on
        train_dataset_name: Name of the dataset to train the probe on
        probe_type: Type of probe to train (default: pytorch_per_token_probe)
        max_samples: Maximum number of samples to evaluate on. If None, uses all samples.
        batch_size: Batch size for evaluation
        output_dir: Directory to save results to. If None, creates a date-stamped directory in DATA_DIR/results

    Returns:
        Tuple of (baseline_results, probe_results)
    """
    # Create output directory
    if output_dir is None:
        date = datetime.now().strftime("%Y%m%d")
        output_dir = DATA_DIR / "results" / f"monitoring_cascade_{date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")

    # Get dataset paths
    eval_dataset_path = EVAL_DATASETS[dataset_name]

    # Load and sample datasets once to ensure consistency
    print("\nLoading datasets...")
    train_dataset = load_dataset(
        dataset_path=train_dataset_path,
        model_name=probe_model_name,
        layer=probe_layer,
        compute_activations=compute_activations,
        n_per_class=max_samples // 2 if max_samples else None,
    )
    eval_dataset = load_dataset(
        dataset_path=eval_dataset_path,
        model_name=probe_model_name,
        layer=probe_layer,
        compute_activations=compute_activations,
        n_per_class=max_samples // 2 if max_samples else None,
    )

    assert eval_dataset.other_fields["scale_labels"] is not None

    # Run baselines for each model
    baseline_results = []
    for model_name in model_names:
        print(f"\nRunning baseline for model: {model_name}")
        # Get model-specific baseline prompt
        baseline_prompt = get_model_baseline_prompt(model_name)
        print(f"Using baseline prompt: {baseline_prompt}")

        # Load model
        model = LLMModel.load(LOCAL_MODELS[model_name])

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

    # Create probe specification
    probe_spec = ProbeSpec(
        name=probe_type,
        hyperparams={
            "batch_size": batch_size,
            "epochs": 10,
            "device": global_settings.DEVICE,
            "optimizer_args": {"lr": 1e-3, "weight_decay": 0.01},
        },
    )

    # Train probe
    print("\nTraining probe...")
    probe = ProbeFactory.build(
        probe=probe_spec,
        train_dataset=train_dataset,
    )

    # Evaluate probe
    print("\nEvaluating probe...")
    probe_scores = probe.predict_proba(eval_dataset)  # .tolist()
    dataset_results = DatasetResults(
        layer=probe_layer,
        metrics=calculate_metrics(eval_dataset.labels_numpy(), probe_scores, fpr=0.01),
    )

    # Get the token counts from the activations array
    # token_counts = eval_dataset.other_fields["activations"].shape[1]
    token_counts = eval_dataset.other_fields["attention_mask"].sum(axis=1)

    # Create EvaluationResult with all scores and labels
    probe_results = EvaluationResult(
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
        token_counts=token_counts.tolist(),
        ids=list(eval_dataset.ids),
        ground_truth_scale_labels=list(eval_dataset.other_fields["scale_labels"]),  # type: ignore
    )

    # Save probe results
    probe_results.save_to(output_dir / "probe_results.jsonl")

    return baseline_results, probe_results


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


def evaluate_probe_baseline_cascade(
    baseline_results: LikelihoodBaselineResults,
    probe_results: EvaluationResult,
    fraction_of_samples: float = 0.2,
    selection_strategy: str = "top",
    remaining_strategy: str = "fixed_0",
) -> CascadeResults:
    assert probe_results.output_scores is not None
    assert probe_results.ground_truth_labels is not None
    assert probe_results.ground_truth_scale_labels is not None
    assert probe_results.token_counts is not None
    assert baseline_results.token_counts is not None

    assert probe_results.ids == baseline_results.ids

    # Calculate number of samples to use
    total_samples = len(probe_results.output_scores)
    num_samples_to_use = int(total_samples * fraction_of_samples)

    # Get indices based on selection strategy
    sorted_indices = np.argsort(probe_results.output_scores)
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

    # Get model-specific per_token_flops
    per_token_flops = get_per_token_flops(baseline_results.model_name)

    # Process top scoring subset
    top_flops = [
        int(per_token_flops) * baseline_results.token_counts[i]
        for i in selected_indices
    ]
    top_results = CascadeResults(
        scores=[baseline_results.high_stakes_scores[i] for i in selected_indices],
        labels=[baseline_results.labels[i] for i in selected_indices],
        ground_truth_labels=[
            baseline_results.ground_truth[i] for i in selected_indices
        ],
        ground_truth_scores=[
            baseline_results.ground_truth_scale_labels[i] for i in selected_indices
        ],
        flops=top_flops,
    )

    if len(remaining_indices) < 1:
        return top_results

    if remaining_strategy.startswith("fixed_"):
        fixed_score = float(remaining_strategy.split("_")[1])
        fixed_label = fixed_score > 0.5

        # Process remaining subset with fixed cascade
        remaining_results = evaluate_fixed_cascade(
            ground_truth_labels=[
                probe_results.ground_truth_labels[i] for i in remaining_indices
            ],
            ground_truth_scale_labels=[
                probe_results.ground_truth_scale_labels[i] for i in remaining_indices
            ],
            fixed_score=fixed_score,
            fixed_label=fixed_label,
        )
    elif remaining_strategy == "probe":
        # Process remaining subset with probe
        all_results = evaluate_probe_baseline_cascade(
            baseline_results=baseline_results,
            probe_results=probe_results,
            fraction_of_samples=fraction_of_samples,
        )
        remaining_results = CascadeResults(
            scores=[all_results.scores[i] for i in remaining_indices],
            labels=[all_results.labels[i] for i in remaining_indices],
            ground_truth_labels=[
                all_results.ground_truth_labels[i] for i in remaining_indices
            ],
            ground_truth_scores=[
                all_results.ground_truth_scores[i] for i in remaining_indices
            ],
            flops=[all_results.flops[i] for i in remaining_indices],
        )
    else:
        raise ValueError(f"Unknown remaining strategy: {remaining_strategy}")

    # Combine results
    return top_results + remaining_results


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
    if evaluation_results.config.probe_spec.name == "pytorch_per_entry_probe_mean":
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
    fixed_score: float = 0.0,
    fixed_label: int = 0,
) -> CascadeResults:
    num_samples = len(ground_truth_labels)
    labels = [fixed_label] * num_samples
    scores = [fixed_score] * num_samples

    return CascadeResults(
        scores=scores,
        labels=labels,
        ground_truth_labels=ground_truth_labels,
        ground_truth_scores=ground_truth_scale_labels,
        flops=[0] * num_samples,  # No computation
    )


if __name__ == "__main__":
    model_names = ["llama-1b", "gemma-1b"]
    dataset_name = "anthropic"
    compute = False
    max_samples = 100

    if compute:
        baseline_results, probe_results = run_monitoring_cascade(
            model_names=model_names,
            dataset_name=dataset_name,
            train_dataset_path=SYNTHETIC_DATASET_PATH / "train.jsonl",
            max_samples=max_samples,
            probe_model_name=LOCAL_MODELS["llama-1b"],
            probe_layer=11,
            compute_activations=True,
        )
    else:
        output_dir = DATA_DIR / "results" / "monitoring_cascade_20250424"
        baseline_results = []
        for model_name in model_names:
            with open(output_dir / f"baseline_{model_name}.jsonl") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        baseline_results.append(
                            LikelihoodBaselineResults.model_validate_json(line.strip())
                        )

        with open(output_dir / "probe_results.jsonl") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    probe_results = EvaluationResult.model_validate_json(line.strip())
                    break  # Assuming we only need the first result

    # Print results
    print("\nBaseline Results:")
    for result in baseline_results:
        for fraction_of_samples in [0.5, 0.75, 1.0]:
            print(f"Model: {result.model_name}, Fraction: {fraction_of_samples}")
            cascade_results = evaluate_single_baseline_cascade(
                result, fraction_of_samples=fraction_of_samples
            )
            print(f"- AUROC: {cascade_results.auroc:.3f}")
            print(f"- Total FLOPs: {sum(cascade_results.flops)}")

    print("\nProbe Results:")
    cascade_results = evaluate_single_probe_cascade(probe_results)
    print(f"- AUROC: {cascade_results.auroc:.3f}")
    print(f"- Total FLOPs: {sum(cascade_results.flops)}")

    # Evaluate probe baseline cascade
    strategies = [
        {"selection_strategy": "top", "remaining_strategy": "fixed_0"},
        {"selection_strategy": "top", "remaining_strategy": "probe"},
        {"selection_strategy": "bottom", "remaining_strategy": "probe"},
        {"selection_strategy": "mid", "remaining_strategy": "probe"},
    ]
    for fraction_of_samples in [0.2, 0.5]:
        for strategy in strategies:
            baseline_result = baseline_results[0]
            print(
                f"\nCascade with fraction of samples: {fraction_of_samples} (baseline model: {baseline_result.model_name})"
            )
            print(f"Strategy: {strategy}")
            probe_baseline_cascade_results = evaluate_probe_baseline_cascade(
                baseline_results=baseline_result,
                probe_results=probe_results,
                fraction_of_samples=fraction_of_samples,
                selection_strategy=strategy["selection_strategy"],
                remaining_strategy=strategy["remaining_strategy"],
            )
            print(f"- AUROC: {probe_baseline_cascade_results.auroc:.3f}")
            print(f"- Total FLOPs: {sum(probe_baseline_cascade_results.flops)}")

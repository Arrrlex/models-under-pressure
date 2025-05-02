import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from models_under_pressure.config import (
    ChooseLayerConfig,
    DataEfficiencyBaselineConfig,
    DataEfficiencyConfig,
    DevSplitFineTuningConfig,
    EvalRunConfig,
    HeatmapRunConfig,
)
from models_under_pressure.interfaces.probes import ProbeSpec


class CVIntermediateResults(BaseModel):
    config: ChooseLayerConfig
    layer_results: dict[int, list[float]] = Field(default_factory=dict)
    layer_mean_accuracies: dict[int, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def add_layer_results(self, layer: int, results: list[float]):
        self.layer_results[layer] = results
        self.layer_mean_accuracies[layer] = float(np.mean(results))

    def save(self):
        self.config.temp_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving intermediate results to {self.config.temp_output_path}")
        with open(self.config.temp_output_path, "a") as f:
            f.write(self.model_dump_json() + "\n")


class CVFinalResults(BaseModel):
    results: CVIntermediateResults
    best_layer: int
    best_layer_accuracy: float

    @classmethod
    def from_intermediate(cls, intermediate: CVIntermediateResults) -> Self:
        best_layer = max(
            intermediate.layer_mean_accuracies.keys(),
            key=lambda x: intermediate.layer_mean_accuracies[x],
        )
        best_layer_accuracy = intermediate.layer_mean_accuracies[best_layer]

        return cls(
            results=intermediate,
            best_layer=best_layer,
            best_layer_accuracy=best_layer_accuracy,
        )

    def save(self):
        path = self.results.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving final results to {path}")
        with open(path, "a") as f:
            f.write(self.model_dump_json() + "\n")


class DatasetResults(BaseModel):
    layer: int
    metrics: dict[str, float]
    """AUROC and other metric scores for each evaluated dataset"""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def save_to(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


class EvaluationResult(BaseModel):
    """Results from evaluating a model on a dataset."""

    config: EvalRunConfig
    """Configuration for the evaluation"""

    dataset_name: str
    """Name of the dataset that was evaluated on"""

    dataset_path: Path
    """Path to the dataset that was evaluated on"""

    metrics: DatasetResults
    """Global metrics for the evaluated dataset"""

    method: str
    """Method used to make predictions"""

    best_epoch: int | None = None
    """Number of epochs used for training (in case validation dataset was used)"""

    output_scores: list[float] | None = None
    """Scores for each example in the eval dataset"""

    output_labels: list[int] | None = None
    """Labels for each example in the eval dataset"""

    ground_truth_labels: list[int] | None = None
    """Ground truth labels for each example in the eval dataset"""

    ground_truth_scale_labels: list[int] | None = None
    """Ground truth scale labels for each example in the eval dataset"""

    token_counts: list[int] | None = None
    """Number of tokens in each sample in the eval dataset"""

    ids: list[str] | None = None
    """Ground truth sample IDs for each example in the eval dataset"""

    mean_of_masked_activations: list[Any] | None = None
    """Mean of the masked activations for each example in the eval dataset"""

    masked_activations: list[Any] | None = None
    """Masked activations for each example in the eval dataset"""

    timestamp: datetime = Field(default_factory=datetime.now)

    def save_to(self, path: Path) -> None:
        with open(path, "a") as f:
            f.write(self.model_dump_json() + "\n")


class BaselineResults(BaseModel):
    ids: list[str]
    accuracy: float
    labels: list[int]
    ground_truth: list[int]
    ground_truth_scale_labels: list[int] | None
    dataset_name: str
    dataset_path: Path
    model_name: str
    max_samples: int | None

    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def save_to(self, path: Path) -> None:
        with open(path, "a") as f:
            f.write(self.model_dump_json() + "\n")


class ContinuationBaselineResults(BaselineResults):
    full_response: list[str]
    valid_response: list[bool]


class ContinuationPrompt(BaseModel):
    high_stakes_completion: str
    low_stakes_completion: str
    system_prompt: str | None = None
    user_prompt: str = "{conversation}"
    conversation_input_key: str = "user_prompt"


class LikelihoodBaselineResults(BaselineResults):
    high_stakes_scores: list[float]
    low_stakes_scores: list[float]
    high_stakes_log_likelihoods: list[float]
    low_stakes_log_likelihoods: list[float]
    prompt_config: ContinuationPrompt
    token_counts: list[int] | None = None


class FinetunedBaselineResults(BaselineResults):
    scores: list[float]
    token_counts: list[int] | None = None


class DevSplitResult(BaseModel):
    """Results for a single dev split fine-tuning run."""

    config: DevSplitFineTuningConfig
    """Configuration used for the k-shot fine-tuning run"""

    k: int
    """Number of examples used for fine-tuning"""

    metrics: dict[str, float]
    """Metrics for the evaluation"""

    probe_scores: list[float]
    """Scores for each example in the eval dataset"""

    ground_truth_labels: list[int]
    """Ground truth labels for each example in the eval dataset"""

    ground_truth_scale_labels: list[int] | None = None
    """Ground truth scale labels for each example in the eval dataset"""

    dataset_name: str
    """Name of the dataset that was evaluated on"""

    dataset_path: Path
    """Path to the dataset that was evaluated on"""

    method: str
    """Method used to make predictions (initial_probe or fine_tuned_probe)"""

    timestamp: datetime = Field(default_factory=datetime.now)

    def save_to(self, path: Path) -> None:
        """Save the results to a file."""
        with open(path, "a") as f:
            f.write(self.model_dump_json() + "\n")


class HeatmapCellResult(BaseModel):
    variation_type: str
    train_variation_value: str
    test_variation_value: str
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d.pop("metrics")
        return d | self.metrics


class HeatmapRunResults(BaseModel):
    config: HeatmapRunConfig
    results: list[HeatmapCellResult]

    def as_pandas(self) -> pd.DataFrame:
        return pd.DataFrame([result.to_dict() for result in self.results])

    def heatmaps(self) -> dict[str, pd.DataFrame]:
        df = self.as_pandas()
        variation_types = df["variation_type"].unique()
        return {
            variation_type: df[df["variation_type"] == variation_type]
            for variation_type in variation_types
        }


class ProbeDataEfficiencyResults(BaseModel):
    probe: ProbeSpec
    dataset_size: int
    metrics: dict[str, float]


class DataEfficiencyResults(BaseModel):
    config: DataEfficiencyConfig
    baseline_config: Optional[DataEfficiencyBaselineConfig] = None
    probe_results: list[ProbeDataEfficiencyResults]
    timestamp: datetime = Field(default_factory=datetime.now)

    def save_to(self, path: Path) -> None:
        with open(path, "a") as f:
            f.write(self.model_dump_json() + "\n")

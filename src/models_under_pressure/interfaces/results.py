import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Self

import numpy as np
from deprecated import deprecated
from pydantic import BaseModel, Field

from models_under_pressure.config import ChooseLayerConfig, EvalRunConfig


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

    output_scores: list[float] | None = None
    """Scores for each example in the eval dataset"""

    output_labels: list[int] | None = None
    """Labels for each example in the eval dataset"""

    ground_truth_labels: list[int] | None = None
    """Ground truth labels for each example in the eval dataset"""

    ground_truth_scale_labels: list[int] | None = None
    """Ground truth scale labels for each example in the eval dataset"""

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


class LikelihoodBaselineResults(BaselineResults):
    high_stakes_scores: list[float]
    low_stakes_scores: list[float]
    high_stakes_log_likelihoods: list[float]
    low_stakes_log_likelihoods: list[float]


@deprecated("Use EvaluationResult instead")
class ProbeEvaluationResults(BaseModel):
    """Results from evaluating probes across multiple datasets."""

    datasets: List[str]
    """Names of the datasets that were evaluated"""

    model_name: str
    """Name of the model that was evaluated"""

    train_dataset_path: str
    """Path to the dataset used to train the probe (str format since Path is not JSON serializable)"""

    metrics: list[DatasetResults]
    """Global metrics for each evaluated dataset"""

    variation_type: Optional[str] = None
    """Type of variation used in training data filtering, if any"""

    variation_value: Optional[str] = None
    """Specific variation value used in training data filtering, if any"""

    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def run_name(self) -> str:
        """Extract metadata and create a run name string.

        Returns:
            String containing layer and variation type info for the run
        """
        run_name = "layer=" + str(self.metrics[0].layer)
        if self.variation_type is not None:
            run_name += ",variation_type=" + self.variation_type
        if self.variation_value is not None:
            run_name += ",variation_value=" + self.variation_value
        return run_name

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def save_to(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


@dataclass
class HeatmapResults:
    probe: str
    performances: Dict[int, np.ndarray]  # Layer -> performance matrix
    variation_values: List[str]  # Values of the variation type
    variation_type: str
    model_name: str
    layers: List[int]
    max_samples: int | None

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "performances": {
                layer: perf.tolist() for layer, perf in self.performances.items()
            },
            "variation_values": self.variation_values,
            "variation_type": self.variation_type,
            "model_name": self.model_name,
            "layers": self.layers,
            "max_samples": self.max_samples,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeatmapResults":
        # Convert performance lists back to numpy arrays
        performances = {
            int(layer): np.array(perf) for layer, perf in data["performances"].items()
        }
        return cls(
            probe=data["probe"],
            performances=performances,
            variation_values=data["variation_values"],
            variation_type=data["variation_type"],
            model_name=data["model_name"],
            layers=data["layers"],
            max_samples=data["max_samples"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

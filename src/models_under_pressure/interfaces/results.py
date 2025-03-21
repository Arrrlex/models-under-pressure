import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from deprecated import deprecated
from pydantic import BaseModel

from models_under_pressure.config import EvalRunConfig


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
    """Name of the dataset that was evaluated"""

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

    def save_to(self, path: Path) -> None:
        with open(path, "a") as f:
            json.dump(self.model_dump(), f)
            f.write("\n")


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
        )

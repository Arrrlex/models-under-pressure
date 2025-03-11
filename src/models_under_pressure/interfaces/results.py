import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ProbeEvaluationResults:
    """Results from evaluating probes across multiple datasets."""

    AUROC: dict[str, float]
    """AUROC scores for each evaluated dataset"""

    datasets: List[str]
    """Names of the datasets that were evaluated"""

    model_name: str
    """Name of the model that was evaluated"""

    layer: int
    """Layer number that was probed"""

    train_dataset_path: str
    """Path to the dataset used to train the probe (str format since Path is not JSON serializable)"""

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
        run_name = "layer=" + str(self.layer)
        if self.variation_type is not None:
            run_name += ",variation_type=" + self.variation_type
        if self.variation_value is not None:
            run_name += ",variation_value=" + self.variation_value
        return run_name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbeEvaluationResults":
        """Create a ProbeEvaluationResults instance from a dictionary.

        Args:
            data: Dictionary containing the serialized ProbeEvaluationResults data

        Returns:
            A new ProbeEvaluationResults instance
        """
        return cls(
            AUROC=data["AUROC"],
            datasets=data["datasets"],
            model_name=data["model_name"],
            layer=data["layer"],
            variation_type=data.get("variation_type"),
            variation_value=data.get("variation_value"),
            train_dataset_path=data["train_dataset_path"],
        )


@dataclass
class HeatmapResults:
    performances: Dict[int, np.ndarray]  # Layer -> performance matrix
    variation_values: List[str]  # Values of the variation type
    variation_type: str
    model_name: str
    layers: List[int]
    max_samples: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
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
            performances=performances,
            variation_values=data["variation_values"],
            variation_type=data["variation_type"],
            model_name=data["model_name"],
            layers=data["layers"],
            max_samples=data["max_samples"],
        )


if __name__ == "__main__":
    # Checking if things are JSON serializable
    results = ProbeEvaluationResults(
        AUROC={"Sandbagging": 0.5, "Deception": 0.6},
        datasets=["Sandbagging", "Deception"],
        model_name="gpt-4o",
        layer=11,
        train_dataset_path="data/results/prompts_28_02_25.jsonl",
    )
    print(json.dumps(dataclasses.asdict(results)))

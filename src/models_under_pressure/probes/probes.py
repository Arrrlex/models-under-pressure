from pathlib import Path
from typing import Optional

from models_under_pressure.interfaces.activations import Aggregator
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import Probe, SklearnProbe


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe: str,
        model: LLMModel,
        train_dataset: LabelledDataset,
        layer: int,
        output_dir: Path,
        aggregator: Optional[Aggregator] = None,
    ) -> Probe:
        if probe == "sklearn_probe":
            assert (
                aggregator is not None
            ), f"aggregator: {aggregator} is required for sklearn probe"
            return SklearnProbe(_llm=model, layer=layer, aggregator=aggregator).fit(
                train_dataset
            )
        elif probe == "pytorch_per_token_probe":
            return PytorchProbe(_llm=model, layer=layer).fit(train_dataset)
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")

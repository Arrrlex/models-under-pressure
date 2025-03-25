from pathlib import Path

from models_under_pressure.interfaces.activations import Aggregator
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import Probe, SklearnProbe
from models_under_pressure.probes.sklearn_probes import Preprocessors, Postprocessors


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe: str,
        model: LLMModel,
        train_dataset: LabelledDataset,
        layer: int,
        output_dir: Path | None = None,
    ) -> Probe:
        if probe == "sklearn_mean_acts":
            return SklearnProbe(
                _llm=model,
                layer=layer,
                aggregator=Aggregator(Preprocessors.mean, Postprocessors.sigmoid),
            ).fit(train_dataset)
        elif probe == "pytorch_per_token":
            return PytorchProbe(_llm=model, layer=layer).fit(train_dataset)
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")

from pathlib import Path

from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchDifferenceOfMeansClassifier,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import (
    Probe,
    SklearnProbe,
)

from pydantic import BaseModel


class ProbeSpec(BaseModel):
    probe_type: str
    layer: int


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe: str,
        model: LLMModel,
        train_dataset: LabelledDataset,
        layer: int,
        output_dir: Path,
        hyper_params: Optional[dict] = None,
    ) -> Probe:
        if probe == "sklearn_mean_agg_probe":
            aggregator=Aggregator(
                preprocessor=Preprocessors.mean,
                postprocessor=Postprocessors.sigmoid,
            )
            if hyper_params is not None:
                return SklearnProbe(
                    _llm=model,
                    layer=layer,
                    aggregator=aggregator,
                    hyper_params=hyper_params,
                ).fit(train_dataset)
            else:
                return SklearnProbe(_llm=model, layer=layer, aggregator=aggregator).fit(
                    train_dataset
                )
        elif probe == "difference_of_means":
            return PytorchProbe(
                _llm=model,
                layer=layer,
                _classifier=PytorchDifferenceOfMeansClassifier(use_lda=False),
            ).fit(train_dataset)
        elif probe == "lda":
            return PytorchProbe(
                _llm=model,
                layer=layer,
                _classifier=PytorchDifferenceOfMeansClassifier(use_lda=True),
            ).fit(train_dataset)
        elif probe == "pytorch_per_token_probe":
            return PytorchProbe(_llm=model, layer=layer).fit(train_dataset)
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")

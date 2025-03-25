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
from models_under_pressure.probes.sklearn_probes import Probe, SklearnProbe
from models_under_pressure.interfaces.probes import ProbeSpec


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe: str | ProbeSpec,
        model: LLMModel,
        train_dataset: LabelledDataset,
        layer: int,
    ) -> Probe:
        if isinstance(probe, str):
            probe = ProbeSpec(name=probe)

        if probe.name == "sklearn_mean_agg_probe":
            assert probe.preprocessor is not None
            assert probe.postprocessor is not None
            return SklearnProbe(
                _llm=model,
                layer=layer,
                aggregator=Aggregator(
                    preprocessor=getattr(Preprocessors, probe.preprocessor),
                    postprocessor=getattr(Postprocessors, probe.postprocessor),
                ),
            ).fit(train_dataset)
        elif probe.name == "difference_of_means":
            return PytorchProbe(
                _llm=model,
                layer=layer,
                _classifier=PytorchDifferenceOfMeansClassifier(use_lda=False),
            ).fit(train_dataset)
        elif probe.name == "lda":
            return PytorchProbe(
                _llm=model,
                layer=layer,
                _classifier=PytorchDifferenceOfMeansClassifier(use_lda=True),
            ).fit(train_dataset)
        elif probe.name == "pytorch_per_token_probe":
            return PytorchProbe(_llm=model, layer=layer).fit(train_dataset)
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")

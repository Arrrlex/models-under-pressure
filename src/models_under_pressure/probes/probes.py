from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchDifferenceOfMeansClassifier,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import Probe, SklearnProbe


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe: str | ProbeSpec,
        model_name: str,
        layer: int,
    ) -> Probe:
        if isinstance(probe, str):
            probe = ProbeSpec(name=probe)

        if probe.name == "sklearn_mean_agg_probe":
            aggregator = Aggregator(
                preprocessor=Preprocessors.mean,
                postprocessor=Postprocessors.sigmoid,
            )
            if probe.hyperparams is not None:
                return SklearnProbe(
                    model_name=model_name,
                    layer=layer,
                    aggregator=aggregator,
                    hyper_params=probe.hyperparams,
                )
            else:
                return SklearnProbe(
                    model_name=model_name,
                    layer=layer,
                    aggregator=aggregator,
                )
        elif probe.name == "difference_of_means":
            assert probe.hyperparams is not None
            return PytorchProbe(
                model_name=model_name,
                layer=layer,
                hyper_params=probe.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=False, training_args=probe.hyperparams
                ),
            )
        elif probe.name == "lda":
            assert probe.hyperparams is not None
            return PytorchProbe(
                model_name=model_name,
                layer=layer,
                hyper_params=probe.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=True, training_args=probe.hyperparams
                ),
            )
        elif probe.name == "pytorch_per_token_probe":
            assert probe.hyperparams is not None
            return PytorchProbe(
                model_name=model_name,
                layer=layer,
                hyper_params=probe.hyperparams,
            )
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")

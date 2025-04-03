from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
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
        probe_spec: str | ProbeSpec,
        model_name: str,
        layer: int,
        train_dataset: LabelledDataset,
    ) -> Probe:
        if isinstance(probe_spec, str):
            probe_spec = ProbeSpec(name=probe_spec)

        if probe_spec.name == "sklearn_mean_agg_probe":
            aggregator = Aggregator(
                preprocessor=Preprocessors.mean,
                postprocessor=Postprocessors.sigmoid,
            )
            if probe_spec.hyperparams is not None:
                probe = SklearnProbe(
                    model_name=model_name,
                    layer=layer,
                    aggregator=aggregator,
                    hyper_params=probe_spec.hyperparams,
                )
            else:
                probe = SklearnProbe(
                    model_name=model_name,
                    layer=layer,
                    aggregator=aggregator,
                )
        elif probe_spec.name == "difference_of_means":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                model_name=model_name,
                layer=layer,
                hyper_params=probe_spec.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=False, training_args=probe_spec.hyperparams
                ),
            )
        elif probe_spec.name == "lda":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                model_name=model_name,
                layer=layer,
                hyper_params=probe_spec.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=True, training_args=probe_spec.hyperparams
                ),
            )
        elif probe_spec.name == "pytorch_per_token_probe":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                model_name=model_name,
                layer=layer,
                hyper_params=probe_spec.hyperparams,
            )
        else:
            raise NotImplementedError(f"Probe type {probe_spec} not supported")

        return probe.fit(train_dataset)

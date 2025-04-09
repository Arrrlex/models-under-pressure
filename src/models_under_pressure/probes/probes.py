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
        probe: str | ProbeSpec,
        train_dataset: LabelledDataset,
    ) -> Probe:
        if not {"activations", "attention_mask", "input_ids"}.issubset(
            train_dataset.other_fields
        ):
            raise ValueError(
                "Dataset must contain activations, attention_mask, and input_ids"
            )

        if isinstance(probe, str):
            probe = ProbeSpec(name=probe)

        if probe.name == "sklearn_mean_agg_probe":
            aggregator = Aggregator(
                preprocessor=Preprocessors.mean,
                postprocessor=Postprocessors.sigmoid,
            )
            if probe.hyperparams is not None:
                return SklearnProbe(
                    aggregator=aggregator,
                    hyper_params=probe.hyperparams,
                ).fit(train_dataset)
            else:
                return SklearnProbe(aggregator=aggregator).fit(train_dataset)
        elif probe.name == "difference_of_means":
            assert probe.hyperparams is not None
            return PytorchProbe(
                hyper_params=probe.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=False, training_args=probe.hyperparams
                ),
            ).fit(train_dataset)
        elif probe.name == "lda":
            assert probe.hyperparams is not None
            return PytorchProbe(
                hyper_params=probe.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=True, training_args=probe.hyperparams
                ),
            ).fit(train_dataset)
        elif probe.name == "pytorch_per_token_probe":
            assert probe.hyperparams is not None
            return PytorchProbe(
                hyper_params=probe.hyperparams,
            ).fit(train_dataset)
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")

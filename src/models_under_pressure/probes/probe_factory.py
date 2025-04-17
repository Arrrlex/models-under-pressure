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
        validation_dataset: LabelledDataset | None = None,
    ) -> Probe:
        if not {"activations", "attention_mask", "input_ids"}.issubset(
            train_dataset.other_fields
        ):
            raise ValueError(
                "Train dataset must contain activations, attention_mask, and input_ids"
            )
        if validation_dataset is not None and not {
            "activations",
            "attention_mask",
            "input_ids",
        }.issubset(validation_dataset.other_fields):
            raise ValueError(
                "Validation dataset must contain activations, attention_mask, and input_ids"
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
                if validation_dataset is not None:
                    print("Warning: Validation dataset is not used for sklearn probes.")
                return SklearnProbe(aggregator=aggregator).fit(train_dataset)
        elif probe.name == "difference_of_means":
            assert probe.hyperparams is not None
            if validation_dataset is not None:
                print(
                    "Warning: Validation dataset is not used for difference-of-means probe."
                )
            return PytorchProbe(
                hyper_params=probe.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=False, training_args=probe.hyperparams
                ),
            ).fit(train_dataset)
        elif probe.name == "lda":
            assert probe.hyperparams is not None
            if validation_dataset is not None:
                print("Warning: Validation dataset is not used for LDA probe.")
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
            ).fit(train_dataset, validation_dataset=validation_dataset)
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")

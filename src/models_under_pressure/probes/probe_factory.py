from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec, ProbeType
from models_under_pressure.probes.probe_store import FullProbeSpec, ProbeStore
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchAdamClassifier,
    PytorchDifferenceOfMeansClassifier,
)
from models_under_pressure.probes.pytorch_modules import (
    AttnLite,
    LinearThenLast,
    LinearThenMax,
    LinearThenMean,
    LinearThenRollingMax,
    LinearThenSoftmax,
    MeanThenLinear,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import (
    Probe,
    SklearnProbe,
)


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe_spec: ProbeSpec,
        train_dataset: LabelledDataset,
        model_name: str,
        layer: int,
        validation_dataset: LabelledDataset | None = None,
        use_store: bool = True,
    ) -> Probe:
        if use_store:
            store = ProbeStore()
            full_spec = FullProbeSpec.from_spec(
                probe_spec,
                model_name=model_name,
                layer=layer,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
            )
            if store.exists(full_spec):
                return store.load(full_spec)

        if not has_activations(train_dataset):
            raise ValueError(
                "Train dataset must contain activations, attention_mask, and input_ids"
            )
        if validation_dataset is not None:
            if not has_activations(validation_dataset):
                raise ValueError(
                    "Validation dataset must contain activations, attention_mask, and input_ids"
                )

        match probe_spec.name:
            case ProbeType.sklearn:
                probe = SklearnProbe(hyper_params=probe_spec.hyperparams)
                return probe.fit(train_dataset)

            case ProbeType.difference_of_means:
                classifier = PytorchDifferenceOfMeansClassifier(
                    use_lda=False,
                    training_args=probe_spec.hyperparams,
                )
            case ProbeType.lda:
                classifier = PytorchDifferenceOfMeansClassifier(
                    use_lda=True,
                    training_args=probe_spec.hyperparams,
                )
            case ProbeType.pre_mean:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams,
                    probe_architecture=MeanThenLinear,
                )
            case ProbeType.attention:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams,
                    probe_architecture=AttnLite,
                )
            case ProbeType.linear_then_mean:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams,
                    probe_architecture=LinearThenMean,
                )
            case ProbeType.linear_then_max:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams,
                    probe_architecture=LinearThenMax,
                )
            case ProbeType.linear_then_softmax:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams,
                    probe_architecture=LinearThenSoftmax,
                )
            case ProbeType.linear_then_rolling_max:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams,
                    probe_architecture=LinearThenRollingMax,
                )
            case ProbeType.linear_then_last:
                classifier = PytorchAdamClassifier(
                    training_args=probe_spec.hyperparams,
                    probe_architecture=LinearThenLast,
                )
        probe = PytorchProbe(
            hyper_params=probe_spec.hyperparams,
            _classifier=classifier,
        )

        probe.fit(train_dataset, validation_dataset)
        if use_store:
            store.save(probe, full_spec)

        return probe


def has_activations(dataset: LabelledDataset) -> bool:
    return {"activations", "attention_mask", "input_ids"} <= set(dataset.other_fields)

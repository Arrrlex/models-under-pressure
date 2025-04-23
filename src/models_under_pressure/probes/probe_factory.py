from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec, ProbeType
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchAttentionClassifier,
    PytorchDifferenceOfMeansClassifier,
    PytorchLinearClassifier,
    PytorchPerEntryLinearClassifier,
)
from models_under_pressure.probes.base import Aggregation
from models_under_pressure.probes.aggregations import (
    Last,
    Max,
    MaxOfRollingMean,
    Mean,
    MeanOfTopK,
    MaxOfSentenceMeans,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import (
    Probe,
    SklearnProbe,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from models_under_pressure.probes.probe_store import ProbeStore


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe_spec: ProbeSpec,
        train_dataset: LabelledDataset,
        model_name: str,
        layer: int,
        validation_dataset: LabelledDataset | None = None,
    ) -> Probe:
        store = ProbeStore()

        try:
            probe = store.load(
                probe_spec,
                model_name,
                layer,
                train_dataset,
                validation_dataset,
            )
            return probe
        except FileNotFoundError:
            print(f"Probe {probe_spec.name} not found in store, building from scratch")

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
            case ProbeType.per_entry:
                classifier = PytorchPerEntryLinearClassifier(
                    training_args=probe_spec.hyperparams,
                )
            case ProbeType.difference_of_means:
                classifier = PytorchDifferenceOfMeansClassifier(
                    use_lda=False, training_args=probe_spec.hyperparams
                )
            case ProbeType.lda:
                classifier = PytorchDifferenceOfMeansClassifier(
                    use_lda=True, training_args=probe_spec.hyperparams
                )
            case ProbeType.attention:
                classifier = PytorchAttentionClassifier(
                    training_args=probe_spec.hyperparams,
                )
            case ProbeType.per_token:
                aggregation = get_aggregation(
                    probe_spec.hyperparams["aggregation"], model_name
                )
                classifier = PytorchLinearClassifier(
                    training_args=probe_spec.hyperparams,
                    aggregation=aggregation,
                )
            case _:
                raise NotImplementedError(f"Probe type {probe_spec.name} not supported")

        probe = PytorchProbe(
            hyper_params=probe_spec.hyperparams,
            _classifier=classifier,
        )

        probe.fit(train_dataset, validation_dataset)

        store = ProbeStore()
        store.save(
            probe,
            probe_spec,
            model_name,
            layer,
            train_dataset.hash,
            validation_dataset.hash if validation_dataset is not None else None,
        )

        return probe


def has_activations(dataset: LabelledDataset) -> bool:
    return {"activations", "attention_mask", "input_ids"} <= set(dataset.other_fields)


def get_aggregation(aggregation_spec: str | dict, model_name: str) -> Aggregation:
    if isinstance(aggregation_spec, str):
        name = aggregation_spec
        kwargs = {}
    else:
        name = aggregation_spec["name"]
        kwargs = {k: v for k, v in aggregation_spec.items() if k != "name"}

    match name:
        case "max":
            return Max(**kwargs)
        case "max-of-sentence-means":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return MaxOfSentenceMeans(tokenizer=tokenizer)
        case "mean-of-top-k":
            return MeanOfTopK(k=kwargs["k"])
        case "mean":
            return Mean()
        case "last":
            return Last()
        case "max-of-rolling-mean":
            return MaxOfRollingMean(window_size=kwargs["window_size"])
        case _:
            raise NotImplementedError(f"Aggregation {name} not supported")

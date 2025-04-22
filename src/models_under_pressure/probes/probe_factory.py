from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec, ProbeType
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchAttentionClassifier,
    PytorchDifferenceOfMeansClassifier,
    PytorchLinearClassifier,
    PytorchPerEntryLinearClassifier,
)
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

from models_under_pressure.probes.store import ProbeStore


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
                train_dataset.hash,
                validation_dataset.hash if validation_dataset is not None else None,
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

        if probe_spec.name == ProbeType.sklearn:
            probe = SklearnProbe(hyper_params=probe_spec.hyperparams)
            return probe.fit(train_dataset, validation_dataset)

        match probe_spec.name:
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
            case ProbeType.max:
                classifier = PytorchLinearClassifier(
                    training_args=probe_spec.hyperparams,
                    aggregation_method=Max(),
                )
            case ProbeType.max_of_sentence_means:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                classifier = PytorchLinearClassifier(
                    training_args=probe_spec.hyperparams,
                    aggregation_method=MaxOfSentenceMeans(tokenizer=tokenizer),
                )
            case ProbeType.mean_of_top_k:
                k = probe_spec.hyperparams["k"]
                classifier = PytorchLinearClassifier(
                    training_args=probe_spec.hyperparams,
                    aggregation_method=MeanOfTopK(k=k),
                )
            case ProbeType.max_of_rolling_mean:
                window_size = probe_spec.hyperparams["window_size"]
                classifier = PytorchLinearClassifier(
                    training_args=probe_spec.hyperparams,
                    aggregation_method=MaxOfRollingMean(window_size=window_size),
                )
            case ProbeType.last:
                classifier = PytorchLinearClassifier(
                    training_args=probe_spec.hyperparams,
                    aggregation_method=Last(),
                )
            case ProbeType.mean:
                classifier = PytorchLinearClassifier(
                    training_args=probe_spec.hyperparams,
                    aggregation_method=Mean(),
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


if __name__ == "__main__":
    from models_under_pressure.config import LOCAL_MODELS, SYNTHETIC_DATASET_PATH
    from models_under_pressure.dataset_utils import load_train_test
    from models_under_pressure.interfaces.probes import ProbeSpec

    # Load the synthetic dataset
    train_dataset, test_dataset = load_train_test(
        dataset_path=SYNTHETIC_DATASET_PATH,
        model_name=LOCAL_MODELS["llama-1b"],
        layer=11,
        compute_activations=True,
        n_per_class=200,
    )

    # Define probe hyperparameters
    probe_spec = ProbeSpec(
        type=ProbeType.per_entry,
        hyperparams={
            "batch_size": 32,
            "epochs": 20,
            "device": "cpu",
            "learning_rate": 1e-2,
            "weight_decay": 0.1,
        },
    )

    # Train the probe
    probe = ProbeFactory.build(
        probe_spec=probe_spec,
        train_dataset=train_dataset,
        validation_dataset=test_dataset,
    )

    def accuracy(probe: Probe, dataset: LabelledDataset) -> float:
        pred_labels = probe.predict_proba(dataset) > 0.5
        return (pred_labels == dataset.labels_numpy()).mean()

    # Print probe performance
    print("Probe training completed!")
    print(f"Train accuracy: {accuracy(probe, train_dataset)}")
    print(f"Test accuracy: {accuracy(probe, test_dataset)}")

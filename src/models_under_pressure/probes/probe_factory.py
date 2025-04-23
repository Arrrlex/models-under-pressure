from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchAttentionClassifier,
    PytorchDifferenceOfMeansClassifier,
    PytorchPerEntryLinearClassifier,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import (
    Probe,
    SklearnProbe,
    compute_accuracy,
)
from models_under_pressure.probes.probe_store import FullProbeSpec, ProbeStore


class ProbeFactory:
    @classmethod
    def build(
        cls,
        probe_spec: ProbeSpec,
        train_dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None,
        model_name: str,
        layer: int,
    ) -> Probe:
        store = ProbeStore()
        full_spec = FullProbeSpec.from_spec(
            spec=probe_spec,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            model_name=model_name,
            layer=layer,
        )
        if store.exists(full_spec):
            return store.load(full_spec)

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

        if (validation_dataset is not None) and (
            probe_spec.name
            not in [
                "pytorch_per_entry_probe_mean",
                "pytorch_per_token_probe",
                "pytorch_attention_probe",
            ]
        ):
            print(
                f"Warning: Validation dataset is not used for probe of type {probe_spec.name}."
            )

        if probe_spec.name == "sklearn_mean_agg_probe":
            probe = SklearnProbe(hyper_params=probe_spec.hyperparams).fit(train_dataset)
        elif probe_spec.name == "difference_of_means":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                hyper_params=probe_spec.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=False, training_args=probe_spec.hyperparams
                ),
            ).fit(train_dataset)
        elif probe_spec.name == "lda":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                hyper_params=probe_spec.hyperparams,
                _classifier=PytorchDifferenceOfMeansClassifier(
                    use_lda=True, training_args=probe_spec.hyperparams
                ),
            ).fit(train_dataset)
        elif probe_spec.name == "pytorch_per_token_probe":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                hyper_params=probe_spec.hyperparams,
            ).fit(train_dataset, validation_dataset=validation_dataset)
        elif probe_spec.name == "pytorch_per_entry_probe_mean":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                hyper_params=probe_spec.hyperparams,
                _classifier=PytorchPerEntryLinearClassifier(
                    training_args=probe_spec.hyperparams
                ),
            ).fit(
                train_dataset, validation_dataset=validation_dataset
            )  # Only functionality for this probe atm
        elif probe_spec.name == "pytorch_attention_probe":
            assert probe_spec.hyperparams is not None
            probe = PytorchProbe(
                hyper_params=probe_spec.hyperparams,
                _classifier=PytorchAttentionClassifier(
                    training_args=probe_spec.hyperparams
                ),
            ).fit(train_dataset, validation_dataset=validation_dataset)
        else:
            raise NotImplementedError(f"Probe type {probe_spec.name} not supported")

        store.save(probe, full_spec)
        return probe


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
        name="pytorch_per_token_probe",
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
        model_name=LOCAL_MODELS["llama-1b"],
        layer=11,
    )

    # Print probe performance
    print("Probe training completed!")
    print(f"Train accuracy: {compute_accuracy(probe, train_dataset)}")
    print(f"Test accuracy: {compute_accuracy(probe, test_dataset)}")

from models_under_pressure.interfaces.activations import (
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchDifferenceOfMeansClassifier,
    PytorchPerEntryLinearClassifier,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import (
    Probe,
    SklearnProbe,
    compute_accuracy,
)


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

        # Warn that validation dataset is not used for any probe except pytorch_per_entry_probe_mean
        if (validation_dataset is not None) and (
            probe.name
            not in ["pytorch_per_entry_probe_mean", "pytorch_per_token_probe"]
        ):
            print("Warning: Validation dataset is not used for LDA probe.")

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
            ).fit(train_dataset, validation_dataset=validation_dataset)
        elif probe.name == "pytorch_per_entry_probe_mean":
            assert probe.hyperparams is not None
            return PytorchProbe(
                hyper_params=probe.hyperparams,
                _classifier=PytorchPerEntryLinearClassifier(
                    training_args=probe.hyperparams
                ),
            ).fit(
                train_dataset, validation_dataset=validation_dataset
            )  # Only functionality for this probe atm
        else:
            raise NotImplementedError(f"Probe type {probe} not supported")


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
        probe=probe_spec, train_dataset=train_dataset, validation_dataset=test_dataset
    )

    # Print probe performance
    print("Probe training completed!")
    print(f"Train accuracy: {compute_accuracy(probe, train_dataset)}")
    print(f"Test accuracy: {compute_accuracy(probe, test_dataset)}")

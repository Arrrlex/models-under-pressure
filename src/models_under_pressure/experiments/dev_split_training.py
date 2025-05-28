import os
from typing import List

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models_under_pressure.config import (
    CONFIG_DIR,
    EVAL_DATASETS,
    EVALUATE_PROBES_DIR,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
    TEST_DATASETS,
    DevSplitFineTuningConfig,
)
from models_under_pressure.dataset_utils import load_dataset, load_splits_lazy
from models_under_pressure.experiments.evaluate_probes import calculate_metrics
from models_under_pressure.interfaces.dataset import (
    LabelledDataset,
    subsample_balanced_subset,
)
from models_under_pressure.interfaces.probes import ProbeSpec, ProbeType
from models_under_pressure.interfaces.results import DatasetResults, DevSplitResult
from models_under_pressure.probes.base import Probe
from models_under_pressure.probes.probe_factory import ProbeFactory


def evaluate_probe(
    probe: Probe,
    eval_dataset: LabelledDataset,
    fpr: float = 0.01,
) -> tuple[List[float], DatasetResults]:
    """Evaluate a probe on a dataset and return scores and metrics."""
    per_entry_probe_scores = probe.predict_proba(eval_dataset)
    y_true = eval_dataset.labels_numpy()
    y_pred = per_entry_probe_scores

    return (
        per_entry_probe_scores.tolist(),  # type: ignore
        DatasetResults(layer=0, metrics=calculate_metrics(y_true, y_pred, fpr)),  # type: ignore
    )


def run_dev_split_fine_tuning(
    config: DevSplitFineTuningConfig,
    use_store: bool = True,
) -> List[DevSplitResult]:
    output_filename = config.output_filename
    if config.evaluate_on_test and not os.path.splitext(output_filename)[0].endswith(
        "_test"
    ):
        # Split the filename and extension
        name, ext = os.path.splitext(output_filename)
        output_filename = f"{name}_test{ext}"

    # Ensure output directory exists (inferred from output path)
    output_path = EVALUATE_PROBES_DIR / output_filename
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)

    # Load and split the training dataset
    splits = load_splits_lazy(
        dataset_path=config.dataset_path,
        dataset_filters=None,
        n_per_class=config.max_samples // 2 if config.max_samples else None,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )
    train_split = splits["train"]
    test_split = splits["test"] if config.validation_dataset else None

    # Create initial probe
    print("Creating initial probe...")
    probe = ProbeFactory.build(
        probe_spec=config.probe_spec,
        model_name=config.model_name,
        layer=config.layer,
        train_dataset=train_split,
        validation_dataset=test_split,
        use_store=use_store,
    )

    # Save the initial probe state
    if hasattr(probe, "_classifier") and hasattr(probe._classifier, "model"):  # type: ignore
        initial_state = probe._classifier.model.state_dict().copy()  # type: ignore
    else:
        initial_state = None

    results_list = []

    if config.eval_dataset_names is None:
        if config.evaluate_on_test:
            eval_dataset_names = list(
                set(EVAL_DATASETS.keys()) & set(TEST_DATASETS.keys())
            )
        else:
            eval_dataset_names = list(EVAL_DATASETS.keys())
    else:
        eval_dataset_names = config.eval_dataset_names

    for eval_dataset_name in tqdm(
        eval_dataset_names, desc="Evaluating on eval datasets"
    ):
        eval_dataset_path = EVAL_DATASETS[eval_dataset_name]
        print(f"Loading eval dataset {eval_dataset_name} from {eval_dataset_path}")
        eval_dataset = load_dataset(
            dataset_path=eval_dataset_path,
            model_name=config.model_name,
            layer=config.layer,
            compute_activations=config.compute_activations,
            n_per_class=config.max_samples // 2 if config.max_samples else None,
        )
        if "scale_labels" not in eval_dataset.other_fields:
            print(
                f"Warning: Skipping {eval_dataset_name} because it does not have scale labels"
            )
            continue
        if config.evaluate_on_test:
            test_dataset_path = TEST_DATASETS[eval_dataset_name]
            print(f"Loading test dataset {eval_dataset_name} from {test_dataset_path}")
            test_dataset = load_dataset(
                dataset_path=test_dataset_path,
                model_name=config.model_name,
                layer=config.layer,
                compute_activations=config.compute_activations,
                n_per_class=config.max_samples // 2 if config.max_samples else None,
            )
            if "scale_labels" not in test_dataset.other_fields:
                print(
                    f"Warning: Skipping {eval_dataset_name} because it does not have scale labels"
                )
                continue
            dev_split = eval_dataset
            test_split = test_dataset
        else:
            # Split eval dataset into train and test
            train_indices, test_indices = train_test_split(
                range(len(eval_dataset)),
                train_size=config.train_split_ratio,
                stratify=eval_dataset.labels_numpy(),
            )

            dev_split = eval_dataset[train_indices]
            test_split = eval_dataset[test_indices]
        del eval_dataset

        # Load initial probe state
        if initial_state is not None:
            probe._classifier.model.load_state_dict(initial_state)  # type: ignore
        else:
            probe = ProbeFactory.build(
                probe_spec=config.probe_spec,
                model_name=config.model_name,
                layer=config.layer,
                train_dataset=train_split,
                validation_dataset=test_split,
                use_store=use_store,
            )

        # Evaluate initial probe on test split
        initial_scores, initial_metrics = evaluate_probe(probe, test_split)
        initial_result = DevSplitResult(
            config=config,
            k=0,
            metrics=initial_metrics.metrics,
            probe_scores=initial_scores,
            ground_truth_labels=test_split.labels_numpy().tolist(),
            ground_truth_scale_labels=test_split.other_fields["scale_labels"],  # type: ignore
            dataset_name=f"{eval_dataset_name}_k0",
            dataset_path=eval_dataset_path,
            method="initial_probe",
        )
        results_list.append(initial_result)
        print(f"Saving initial results to {EVALUATE_PROBES_DIR / output_filename}")
        initial_result.save_to(EVALUATE_PROBES_DIR / output_filename)

        # Fine-tune and evaluate for each k
        for k in tqdm(config.k_values, desc=f"Fine-tuning on {eval_dataset_name}"):
            if k > len(dev_split):
                print(
                    f"Skipping k={k} as it's larger than train split size {len(dev_split)}"
                )
                continue

            # Sample k examples from train split
            k_split = subsample_balanced_subset(dev_split, n_per_class=k // 2)

            if config.dev_sample_usage == "combine":
                # Combine train split and k_split using LabelledDataset.concatenate
                # This will automatically handle field intersection and padding
                combined_split = LabelledDataset.concatenate(
                    [train_split] + [k_split] * config.sample_repeats,
                    col_conflict="intersection",
                )
                probe = ProbeFactory.build(
                    probe_spec=config.probe_spec,
                    model_name=config.model_name,
                    layer=config.layer,
                    train_dataset=combined_split,
                    validation_dataset=None,
                    use_store=use_store,
                )
                del combined_split
            elif config.dev_sample_usage == "only":
                # TODO Unsure if gradient_accumulation has to be set to 1 here
                probe = ProbeFactory.build(
                    probe_spec=config.probe_spec,
                    model_name=config.model_name,
                    layer=config.layer,
                    train_dataset=k_split,
                    validation_dataset=None,
                    use_store=use_store,
                )
            elif config.dev_sample_usage == "fine-tune":
                # Restore initial probe state before fine-tuning
                if (
                    initial_state is not None
                    and hasattr(probe, "_classifier")
                    and hasattr(probe._classifier, "model")  # type: ignore
                ):
                    probe._classifier.model.load_state_dict(initial_state)  # type: ignore
                else:
                    raise NotImplementedError(
                        "Cannot restore initial probe state for this probe type"
                    )

                # Fine-tune probe
                print(f"Fine-tuning probe on {k} examples...")
                if hasattr(probe, "_classifier") and hasattr(
                    probe._classifier,  # type: ignore
                    "training_args",
                ):
                    probe._classifier.training_args["epochs"] = config.fine_tune_epochs  # type: ignore
                    probe._classifier.training_args["gradient_accumulation_steps"] = 1  # type: ignore
                probe.fit(k_split, initialize_model=False)
            else:
                raise ValueError(
                    f"Invalid dev_sample_usage: {config.dev_sample_usage}. Must be one of: 'fine-tune', 'only', 'combine'"
                )

            # Evaluate fine-tuned probe
            fine_tuned_scores, fine_tuned_metrics = evaluate_probe(probe, test_split)
            fine_tuned_result = DevSplitResult(
                config=config,
                k=k,
                metrics=fine_tuned_metrics.metrics,
                probe_scores=fine_tuned_scores,
                ground_truth_labels=test_split.labels_numpy().tolist(),
                ground_truth_scale_labels=test_split.other_fields["scale_labels"],  # type: ignore
                dataset_name=f"{eval_dataset_name}_k{k}",
                dataset_path=eval_dataset_path,
                method="fine_tuned_probe",
            )
            results_list.append(fine_tuned_result)

            print(
                f"Saving finetuned results to {EVALUATE_PROBES_DIR / output_filename}"
            )
            fine_tuned_result.save_to(EVALUATE_PROBES_DIR / output_filename)

        del test_split
        del dev_split

    return results_list


def run_dev_split_fine_tuning_single_k(
    config: DevSplitFineTuningConfig,
    k: int,
    eval_dataset_name: str,
    use_store: bool = True,
) -> List[DevSplitResult]:
    """
    Similar to run_dev_split_fine_tuning, but only runs fine-tuning and evaluation for a single value of k and a single eval dataset.

    This function is optimized for using less working memory, so that things don't crash.
    """
    assert config.evaluate_on_test, "evaluate_on_test must be True for this function!"

    output_filename = config.output_filename
    if config.evaluate_on_test and not os.path.splitext(output_filename)[0].endswith(
        "_test"
    ):
        name, ext = os.path.splitext(output_filename)
        output_filename = f"{name}_test{ext}"

    output_path = EVALUATE_PROBES_DIR / output_filename
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)

    splits = load_splits_lazy(
        dataset_path=config.dataset_path,
        dataset_filters=None,
        n_per_class=config.max_samples // 2 if config.max_samples else None,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
    )
    train_split = splits["train"]
    # test_split = splits["test"] if config.validation_dataset else None

    results_list = []

    # Only run for the provided eval_dataset_name
    eval_dataset_path = EVAL_DATASETS[eval_dataset_name]
    print(f"Loading eval dataset {eval_dataset_name} from {eval_dataset_path}")
    dev_split = load_dataset(
        dataset_path=eval_dataset_path,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
        n_per_class=config.max_samples // 2 if config.max_samples else None,
    )
    if "scale_labels" not in dev_split.other_fields:
        print(
            f"Warning: Skipping {eval_dataset_name} because it does not have scale labels"
        )
        return results_list

    # Only run for the provided k
    if k > len(dev_split):
        print(f"Skipping k={k} as it's larger than train split size {len(dev_split)}")
        return results_list

    # Subsample and overwrite dev_split to save memory
    dev_split = subsample_balanced_subset(dev_split, n_per_class=k // 2)

    if config.dev_sample_usage == "combine":
        print(f"Combining train split and {config.sample_repeats} dev splits")
        # Overwrite train_split to save memory
        train_split.extend_in_place([dev_split] * config.sample_repeats)
        # train_split = LabelledDataset.concatenate(
        #    [train_split] + ([dev_split] * config.sample_repeats),
        #    col_conflict="intersection",
        # )
        del dev_split
        probe = ProbeFactory.build(
            probe_spec=config.probe_spec,
            model_name=config.model_name,
            layer=config.layer,
            train_dataset=train_split,
            validation_dataset=None,
            use_store=use_store,
        )
        del train_split
    else:
        raise ValueError(
            f"Invalid dev_sample_usage: {config.dev_sample_usage}. Must be one of: 'combine'"
        )

    # Only now loading test split to save memory
    test_dataset_path = TEST_DATASETS[eval_dataset_name]
    print(f"Loading test dataset {eval_dataset_name} from {test_dataset_path}")
    test_split = load_dataset(
        dataset_path=test_dataset_path,
        model_name=config.model_name,
        layer=config.layer,
        compute_activations=config.compute_activations,
        n_per_class=config.max_samples // 2 if config.max_samples else None,
    )
    if "scale_labels" not in test_split.other_fields:
        print(
            f"Warning: Skipping {eval_dataset_name} because it does not have scale labels"
        )
        return results_list

    fine_tuned_scores, fine_tuned_metrics = evaluate_probe(probe, test_split)
    fine_tuned_result = DevSplitResult(
        config=config,
        k=k,
        metrics=fine_tuned_metrics.metrics,
        probe_scores=fine_tuned_scores,
        ground_truth_labels=test_split.labels_numpy().tolist(),
        ground_truth_scale_labels=test_split.other_fields["scale_labels"],  # type: ignore
        dataset_name=f"{eval_dataset_name}_k{k}",
        dataset_path=eval_dataset_path,
        method="fine_tuned_probe",
    )
    results_list.append(fine_tuned_result)

    print(f"Saving finetuned results to {EVALUATE_PROBES_DIR / output_filename}")
    fine_tuned_result.save_to(EVALUATE_PROBES_DIR / output_filename)

    del test_split

    return results_list


if __name__ == "__main__":
    evaluate_on_test = True

    probe_name = "attention"  # Set probe name first
    # Load probe config
    probe_type = ProbeType(probe_name)  # Convert string to enum

    # Find the matching probe config file
    probe_config = None
    for config_file in (CONFIG_DIR / "probe").glob("*.yaml"):
        with open(config_file) as f:
            current_config = yaml.safe_load(f)
            if current_config.get("name") == probe_name:
                probe_config = current_config
                break

    if probe_config is None:
        raise ValueError(f"No probe config found for probe name: {probe_name}")

    # Ensure numeric values are properly typed
    if "optimizer_args" in probe_config["hyperparams"]:
        probe_config["hyperparams"]["optimizer_args"]["lr"] = float(
            probe_config["hyperparams"]["optimizer_args"]["lr"]
        )
        probe_config["hyperparams"]["optimizer_args"]["weight_decay"] = float(
            probe_config["hyperparams"]["optimizer_args"]["weight_decay"]
        )
    if "final_lr" in probe_config["hyperparams"]:
        probe_config["hyperparams"]["final_lr"] = float(
            probe_config["hyperparams"]["final_lr"]
        )

    config = DevSplitFineTuningConfig(
        # fine_tune_epochs=10,
        dev_sample_usage="combine",
        fine_tune_epochs=20,
        model_name=LOCAL_MODELS["llama-70b"],
        layer=31,
        max_samples=None,
        compute_activations=False,
        probe_spec=ProbeSpec(
            name=probe_type,
            hyperparams=probe_config["hyperparams"],
        ),
        dataset_path=SYNTHETIC_DATASET_PATH,
        sample_repeats=5,
        validation_dataset=True,
        evaluate_on_test=evaluate_on_test,
        # eval_datasets=[EVAL_DATASETS["anthropic"]],
        eval_dataset_names=["toolace"],
        # k_values=[4, 8, 16, 32, 64],
        # On Anthropic crashes after k=2, on MTS crashes after k=64
        # For MT, immediately crashing, now trying with lower batch size of 64, same!
        # ToolACE immediately crashing
        output_filename="dev_split_training_neurips.jsonl",
    )

    # double_check_config(config)

    for _ in range(5):
        print("Running dev split training experiment")
        print(
            f"Results will be saved to {EVALUATE_PROBES_DIR / config.output_filename}"
        )
        for k in config.k_values:
            for eval_dataset_name in config.eval_dataset_names:
                results = run_dev_split_fine_tuning_single_k(
                    config=config,
                    k=k,
                    eval_dataset_name=eval_dataset_name,
                    use_store=False,
                )
                for result in results:
                    print("-" * 100)
                    print(result.dataset_name)
                    print(result.metrics)
        # results = run_dev_split_fine_tuning(config, use_store=False)
        # for result in results:
        #    print("-" * 100)
        #    print(result.dataset_name)
        #    print(result.metrics)

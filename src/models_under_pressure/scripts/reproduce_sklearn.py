from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import (
    DATA_DIR,
    EVAL_DATASETS,
    LOCAL_MODELS,
    SYNTHETIC_DATASET_PATH,
)
from models_under_pressure.experiments.dataset_splitting import (
    load_filtered_train_dataset,
)
from models_under_pressure.interfaces.activations import (
    Activation,
    Aggregator,
    Postprocessors,
    Preprocessors,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchAttentionClassifier,
)
from models_under_pressure.probes.pytorch_probes import PytorchProbe
from models_under_pressure.probes.sklearn_probes import SklearnProbe


def save_activation(activation: Activation, path: Path) -> None:
    """Save activation object's components as numpy arrays in half precision"""
    np.savez_compressed(
        path,
        activations=activation.get_activations(per_token=False).astype(np.float16),
        attention_mask=activation.get_attention_mask(per_token=False).astype(
            np.float16
        ),
        input_ids=activation.get_input_ids().astype(np.int32),
    )


def load_activation(path: Path) -> Activation:
    """Load activation object from numpy arrays"""
    data = np.load(path)
    return Activation(
        _activations=data["activations"].astype(np.float16),
        _attention_mask=data["attention_mask"].astype(np.float16),
        _input_ids=data["input_ids"].astype(np.int32),
    )


def compute_and_cache_activations(
    model: LLMModel, datasets: Dict[str, LabelledDataset], layer: int, cache_dir: Path
) -> Dict[str, tuple[Activation, np.ndarray]]:
    """Compute and cache activations for each dataset"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    activations_dict = {}

    for name, dataset in datasets.items():
        cache_path = cache_dir / f"{name}_layer{layer}_activations.npz"
        labels_path = cache_dir / f"{name}_labels.npy"

        if cache_path.exists() and labels_path.exists():
            print(f"Loading cached activations for {name}...")
            activations = load_activation(cache_path)
            labels = np.load(labels_path)
        else:
            print(f"Computing activations for {name}...")
            activations = model.get_batched_activations(dataset=dataset, layer=layer)
            labels = dataset.labels_numpy()

            # Cache the results
            save_activation(activations, cache_path)
            np.save(labels_path, labels)

        activations_dict[name] = (activations, labels)

    return activations_dict


def compare_probes(
    train_dataset_path: Path,
    layer: int = 11,
    max_samples: int | None = None,
    cache_dir: Path = DATA_DIR / "temp/probe_comparison",
    eval_dataset_names: list[str] | None = None,
):
    # Set random seeds for reproducibility
    np.random.seed(42)

    # Check if all cached files exist first
    cache_dir.mkdir(parents=True, exist_ok=True)
    eval_names = (
        eval_dataset_names if eval_dataset_names is not None else EVAL_DATASETS.keys()
    )

    all_dataset_names = ["train"] + list(eval_names)
    all_cached = True

    for name in all_dataset_names:
        cache_path = cache_dir / f"{name}_layer{layer}_activations.npz"
        labels_path = cache_dir / f"{name}_labels.npy"
        if not (cache_path.exists() and labels_path.exists()):
            all_cached = False
            break

    # Load cached activations if they all exist
    if all_cached:
        print("All activations found in cache, loading...")
        activations_dict = {
            name: (
                load_activation(cache_dir / f"{name}_layer{layer}_activations.npz"),
                np.load(cache_dir / f"{name}_labels.npy"),
            )
            for name in all_dataset_names
        }
        model = None
    else:
        # Original model and dataset loading logic
        print("Some activations missing, loading model and datasets...")
        model = LLMModel.load(model_name=LOCAL_MODELS["llama-1b"])
        train_dataset = load_filtered_train_dataset(
            dataset_path=train_dataset_path, max_samples=max_samples
        )
        eval_datasets = {
            name: (
                LabelledDataset.load_from(path).sample(max_samples)
                if max_samples is not None
                else LabelledDataset.load_from(path)
            )
            for name, path in EVAL_DATASETS.items()
            if eval_dataset_names is None or name in eval_dataset_names
        }
        datasets = {"train": train_dataset, **eval_datasets}
        activations_dict = compute_and_cache_activations(
            model=model, datasets=datasets, layer=layer, cache_dir=cache_dir
        )

    # Configure sklearn probe
    print("Training sklearn probe...")
    sklearn_aggregator = Aggregator(
        preprocessor=Preprocessors.mean,
        postprocessor=Postprocessors.sigmoid,
    )
    sklearn_probe = SklearnProbe(
        _llm=model,
        layer=layer,
        aggregator=sklearn_aggregator,
        hyper_params={
            # "C": 1e-3,
            "C": 1,
            "random_state": 42,
            "fit_intercept": False,
            # "max_iter": 1,
            # "solver": "liblinear",
            # "solver": "newton-cg",
        },
    )

    # Configure pytorch probe
    print("Training pytorch probe...")
    # previous_args = {
    #     "batch_size": 32,
    #     "epochs": 6,
    #     "device": "cuda" if torch.cuda.is_available() else "cpu",
    #     "learning_rate": 1e-4,
    #     "weight_decay": 0.01,
    # }
    # lbfgs_args = {
    #     "optimizer_type": "lbfgs",
    #     "batch_size": 256,
    #     "epochs": 20,
    #     "device": "cuda" if torch.cuda.is_available() else "cpu",
    #     "learning_rate": 1e-2,
    #     "max_iter": 20,
    #     "line_search_fn": "strong_wolfe",
    #     "weight_decay": 0.01,
    #     "tolerance_change": 1e-4,
    #     # "tolerance_grad": 64 * np.finfo(float).eps,
    # }
    adamw_args = {
        "batch_size": 128,
        "epochs": 40,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "optimizer_args": {"lr": 1e-2, "weight_decay": 0.001},
        # "betas": (0.9, 0.999),  # -> Made things worse to change
        # "differentiable": True,
        # "learning_rate": 1e-3,
        # "weight_decay": 0.001,
        # "amsgrad": True,
    }
    pytorch_probe = PytorchProbe(
        _llm=model,
        layer=layer,
        hyper_params={},  # Doesn't apply when giving the classifier
        _classifier=PytorchAttentionClassifier(
            training_args=adamw_args,
            # training_args=lbfgs_args,
        ),
    )
    # Initial weights shape: (2048,)
    # X shape: (10, 2048)
    # Target shape: (10,)
    # Sample weight: None
    # L2 regularization strength: 100.0
    # Number of threads: 1
    # Max line search steps: 50
    # Tolerance (ftol): 1.4210854715202004e-14
    # DONE:
    # Max iterations: 100
    # Tolerance (gtol): 0.0001

    # Train both probes on training activations
    train_activations, train_labels = activations_dict["train"]
    sklearn_probe._fit(train_activations, train_labels)
    pytorch_probe._fit(train_activations, train_labels)

    # Evaluate on eval datasets
    print("\nEvaluating on eval datasets...")

    for dataset_name in eval_names:
        print(f"\nResults for {dataset_name}:")
        eval_activations, eval_labels = activations_dict[dataset_name]

        # Get predictions from both probes
        sklearn_probs = sklearn_probe._predict_proba(eval_activations)
        pytorch_probs = pytorch_probe._classifier.predict_proba(eval_activations)  # type: ignore

        # Calculate metrics
        sklearn_auroc = roc_auc_score(eval_labels, sklearn_probs)
        pytorch_auroc = roc_auc_score(eval_labels, pytorch_probs)

        print(f"Sklearn AUROC: {sklearn_auroc:.4f}")
        print(f"Pytorch AUROC: {pytorch_auroc:.4f}")

        # Calculate correlation between predictions
        correlation = np.corrcoef(sklearn_probs, pytorch_probs)[0, 1]
        print(f"Correlation between predictions: {correlation:.4f}")


if __name__ == "__main__":
    compare_probes(
        train_dataset_path=SYNTHETIC_DATASET_PATH,
        layer=11,
        max_samples=None,  # Adjust this as needed
    )

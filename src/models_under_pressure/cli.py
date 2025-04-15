import subprocess
import sys
from pathlib import Path

import typer

from models_under_pressure.activation_store import ActivationStore, ActivationsSpec
from models_under_pressure.config import LOCAL_MODELS
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel


def dashboard_command():
    """Run the Streamlit dashboard with any provided arguments."""
    dashboard_path = Path(__file__).parent / "dashboard.py"

    # Get any arguments passed after the dash command
    args = sys.argv[1:]

    # Use Streamlit's Python API to run the dashboard
    subprocess.run(["streamlit", "run", str(dashboard_path), "--"] + args)


activation_store_cli = typer.Typer(pretty_exceptions_show_locals=False)


@activation_store_cli.command()
def store(
    model_name: str = typer.Option(..., "--model", help="Name of the model to use"),
    dataset_path: Path = typer.Option(
        ..., "--dataset", "--datasets", help="Path to the dataset or datasets"
    ),
    layers_str: str = typer.Option(
        ..., "--layers", "--layer", help="Comma-separated list of layer numbers"
    ),
    batch_size: int = typer.Option(4, "--batch", help="Batch size for processing"),
):
    """Calculate and store activations for a model and dataset."""
    layers = _parse_layers(layers_str)
    model_name = _parse_model_name(model_name)
    dataset_paths = _parse_dataset_path(dataset_path)
    print(f"Storing activations for {model_name} on {dataset_paths}")

    store = ActivationStore()

    model = LLMModel.load(model_name, batch_size=batch_size)

    for dataset_path in dataset_paths:
        print(f"Storing activations for {dataset_path}")
        dataset = LabelledDataset.load_from(dataset_path)
        filtered_layers = []
        for layer in layers:
            activations_spec = ActivationsSpec(
                model_name=model_name,
                dataset_path=dataset_path,
                layer=layer,
            )
            if store.exists(activations_spec):
                print(f"Layer {layer} already exists, skipping")
            else:
                filtered_layers.append(layer)

        if not filtered_layers:
            print(f"No layers to store for {dataset_path}")
            continue

        activations, inputs = model.get_batched_activations_for_layers(
            dataset=dataset,
            layers=filtered_layers,
        )

        approx_size = activations.numel() * activations.element_size()
        print(
            f"Approximately {approx_size / 10**9:.2f}GB of activations without compression"
        )

        store.save(model_name, dataset_path, filtered_layers, activations, inputs)

        store.sync()


@activation_store_cli.command()
def delete(
    model_name: str = typer.Option(
        ...,
        "--model",
        help="Name of the model to use",
    ),
    dataset_path: Path = typer.Option(
        ...,
        "--dataset",
        "--datasets",
        help="Path to the dataset or datasets (can include wildcards)",
    ),
    layers_str: str = typer.Option(
        ...,
        "--layer",
        "--layers",
        help="Comma-separated list of layer numbers",
    ),
):
    """Delete activations for a model and dataset."""
    layers = _parse_layers(layers_str)
    model_name = _parse_model_name(model_name)
    dataset_paths = _parse_dataset_path(dataset_path)

    store = ActivationStore()
    for dataset_path in dataset_paths:
        for layer in layers:
            spec = ActivationsSpec(
                model_name=model_name,
                dataset_path=dataset_path,
                layer=layer,
            )
            store.delete(spec)

    store.sync()


@activation_store_cli.command()
def sync():
    """Sync the activation store."""
    store = ActivationStore()
    store.sync()


def _parse_layers(layers_str: str) -> list[int]:
    """Parse a comma-separated list of layer numbers."""
    return [int(layer) for layer in layers_str.split(",")]


def _parse_dataset_path(dataset_path: Path) -> list[Path]:
    """Parse a path to a dataset or datasets.

    Supports both direct paths and wildcard patterns (e.g. data/**/*.csv).
    Can handle both absolute and relative paths.
    """
    if "*" in str(dataset_path):
        return list(Path.cwd().glob(str(dataset_path)))
    else:
        # Handle direct path
        return [dataset_path]


def _parse_model_name(model_name: str) -> str:
    """Parse a model name."""
    return LOCAL_MODELS.get(model_name, model_name)

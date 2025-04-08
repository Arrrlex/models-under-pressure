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
    dataset_path: Path = typer.Option(..., "--dataset", help="Path to the dataset"),
    layers: str = typer.Option(
        ..., "--layers", "--layer", help="Comma-separated list of layer numbers"
    ),
    batch_size: int = typer.Option(32, "--batch", help="Batch size for processing"),
):
    """Calculate and store activations for a model and dataset."""
    layer_list = [int(layer) for layer in layers.split(",")]
    model_name = LOCAL_MODELS.get(model_name, model_name)

    store = ActivationStore()
    filtered_layers = []
    for layer in layer_list:
        activations_spec = ActivationsSpec(
            model_name=model_name,
            dataset_path=dataset_path,
            layer=layer,
        )
        if store.exists(activations_spec):
            print(f"Layer {layer} already exists, skipping")
        else:
            filtered_layers.append(layer)

    model = LLMModel.load(model_name, batch_size=batch_size)
    dataset = LabelledDataset.load_from(dataset_path)

    activations, inputs = model.get_batched_activations_for_layers(
        dataset=dataset,
        layers=filtered_layers,
    )

    approx_size = activations.numel() * activations.element_size()
    print(
        f"Approximately {approx_size / 10**9:.2f}GB of activations without compression"
    )

    store.save(model_name, dataset_path, filtered_layers, activations, inputs)


@activation_store_cli.command()
def push_manifest():
    """Push the current local manifest to the remote bucket."""
    store = ActivationStore()
    store.push_manifest()


@activation_store_cli.command()
def clean():
    """Clean up local and remote activations."""
    store = ActivationStore()
    store.clean()

import subprocess
import sys
from pathlib import Path

import typer

from models_under_pressure.activation_store import ActivationsSpec, ActivationStore
from models_under_pressure.config import LOCAL_MODELS
from models_under_pressure.dataset_store import DatasetStore
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel


def dashboard_command():
    """Run the Streamlit dashboard with any provided arguments."""
    dashboard_path = Path(__file__).parent / "dashboard.py"

    # Get any arguments passed after the dash command
    args = sys.argv[1:]

    # Use Streamlit's Python API to run the dashboard
    subprocess.run(["streamlit", "run", str(dashboard_path), "--"] + args)


class ActivationStoreCLI:
    def __init__(self):
        self.app = typer.Typer(pretty_exceptions_show_locals=False)
        self._register_commands()

    def _register_commands(self):
        @self.app.command()
        def store(
            model_name: str = typer.Option(
                ...,
                "--model",
                help="Name of the model to use",
            ),
            dataset_path: Path = typer.Option(
                ...,
                "--dataset",
                "--datasets",
                help="Path to the dataset or datasets",
            ),
            layers_str: str = typer.Option(
                ...,
                "--layers",
                "--layer",
                help="Comma-separated list of layer numbers",
            ),
            batch_size: int = typer.Option(
                4,
                "--batch",
                help="Batch size for processing",
            ),
        ):
            """Calculate and store activations for a model and dataset."""
            layers = self._parse_layers(layers_str)
            model_name = self._parse_model_name(model_name)
            dataset_paths = self._parse_dataset_path(dataset_path)
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

                store.save(
                    model_name, dataset_path, filtered_layers, activations, inputs
                )

        @self.app.command()
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
            layers = self._parse_layers(layers_str)
            model_name = self._parse_model_name(model_name)
            dataset_paths = self._parse_dataset_path(dataset_path)

            store = ActivationStore()

            specs = [
                ActivationsSpec(
                    model_name=model_name,
                    dataset_path=dataset_path,
                    layer=layer,
                )
                for dataset_path in dataset_paths
                for layer in layers
            ]
            store.sync(add_specs=[], remove_specs=specs)

        @self.app.command()
        def sync():
            """Sync the activation store."""
            store = ActivationStore()
            store.sync(add_specs=[], remove_specs=[])

    def _parse_layers(self, layers: str) -> list[int]:
        """Parse a comma-separated list of layer numbers."""
        return [int(layer) for layer in layers.split(",")]

    def _parse_dataset_path(self, dataset_path: Path) -> list[Path]:
        """Parse a path to a dataset or datasets.

        Supports both direct paths and wildcard patterns (e.g. data/**/*.csv).
        Can handle both absolute and relative paths.
        """
        if "*" in str(dataset_path):
            return list(Path.cwd().glob(str(dataset_path)))
        else:
            # Handle direct path
            return [dataset_path]

    def _parse_model_name(self, model_name: str) -> str:
        """Parse a model name."""
        return LOCAL_MODELS.get(model_name, model_name)


class DatasetStoreCLI:
    def __init__(self):
        self.app = typer.Typer(pretty_exceptions_show_locals=False)
        self._register_commands()

    def _register_commands(self):
        @self.app.command()
        def upload(
            dataset_paths: Path = typer.Argument(
                ...,
                help="Path to the dataset or datasets (can include wildcards)",
            ),
        ):
            """Upload a dataset or datasets."""
            paths = self._parse_dataset_path(dataset_paths)
            store = DatasetStore()
            store.upload(paths)

        @self.app.command()
        def download():
            """Download all datasets from the store."""
            store = DatasetStore()
            store.download_all()

    def _parse_dataset_path(self, dataset_path: Path) -> list[Path]:
        """Parse a path to a dataset or datasets.

        Supports both direct paths and wildcard patterns (e.g. data/**/*.csv).
        Can handle both absolute and relative paths.
        """
        if "*" in str(dataset_path):
            return list(Path.cwd().glob(str(dataset_path)))
        else:
            # Handle direct path
            return [dataset_path]


# Create the main app
app = typer.Typer(pretty_exceptions_show_locals=False)

# Add subcommands
app.add_typer(ActivationStoreCLI().app, name="acts")
app.add_typer(DatasetStoreCLI().app, name="datasets")


# Add dashboard command
@app.command()
def dashboard():
    """Run the Streamlit dashboard with any provided arguments."""
    dashboard_path = Path(__file__).parent / "dashboard.py"
    args = sys.argv[1:]
    subprocess.run(["streamlit", "run", str(dashboard_path), "--"] + args)


@app.command()
def exp(
    experiment_args: list[str] = typer.Argument(
        None, help="Arguments to pass to the experiment script"
    ),
):
    """Run an experiment."""
    run_experiments_path = Path(__file__).parent / "scripts/run_experiment.py"
    subprocess.run(["python", str(run_experiments_path)] + experiment_args)

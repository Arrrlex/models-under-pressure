from pathlib import Path
import typer
from models_under_pressure.activation_store import compute_activations_and_save
from models_under_pressure.config import ACTIVATIONS_DIR, LOCAL_MODELS
from models_under_pressure.interfaces.dataset import DatasetSpec


def main(
    model_name: str = typer.Option(..., "--model"),
    dataset_path: Path = typer.Option(..., "--dataset"),
    layers_str: str = typer.Option(
        ...,
        "--layers",
        help="Comma-separated list of layers to process",
    ),
):
    layers = [int(layer) for layer in layers_str.split(",")]
    compute_activations_and_save(
        model_name=LOCAL_MODELS.get(model_name, model_name),
        dataset_spec=DatasetSpec(path=dataset_path),
        layers=layers,
        activations_dir=ACTIVATIONS_DIR,
    )


if __name__ == "__main__":
    typer.run(main)

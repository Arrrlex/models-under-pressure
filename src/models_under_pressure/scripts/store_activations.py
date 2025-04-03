from pathlib import Path
import typer
from models_under_pressure.activation_store import ActivationStore
from models_under_pressure.config import ACTIVATIONS_DIR, LOCAL_MODELS
from models_under_pressure.interfaces.dataset import DatasetSpec, LabelledDataset
from models_under_pressure.model import LLMModel


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

    model = LLMModel(LOCAL_MODELS.get(model_name, model_name))
    dataset_spec = DatasetSpec(path=dataset_path)
    dataset = LabelledDataset.load_from(dataset_spec)

    activations, inputs = model.get_batched_activations(dataset, layers)

    approx_size = activations.numel() * activations.element_size()
    print(f"Approximately {approx_size / 10**9:.2f}GB of activations")

    store = ActivationStore(ACTIVATIONS_DIR)
    store.save(model.name, dataset_spec, layers, activations, inputs)


if __name__ == "__main__":
    typer.run(main)

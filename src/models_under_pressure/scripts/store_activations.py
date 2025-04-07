from pathlib import Path
import typer
from models_under_pressure.activation_store import ActivationStore, ActivationsSpec
from models_under_pressure.config import ACTIVATIONS_DIR, LOCAL_MODELS
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel


def main(
    model_name: str = typer.Option(..., "--model"),
    dataset_path: Path = typer.Option(..., "--dataset"),
    layers_str: str = typer.Option(
        ...,
        "--layers",
        help="Comma-separated list of layers to process",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size to use for storing activations",
    ),
):
    model_name = LOCAL_MODELS.get(model_name, model_name)
    store = ActivationStore(ACTIVATIONS_DIR)
    layers = [int(layer) for layer in layers_str.split(",")]
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
    layers = filtered_layers

    model = LLMModel.load(model_name, batch_size=batch_size)
    dataset = LabelledDataset.load_from(dataset_path)

    activations, inputs = model.get_batched_activations_for_layers(
        dataset=dataset,
        layers=layers,
    )

    approx_size = activations.numel() * activations.element_size()
    print(
        f"Approximately {approx_size / 10**9:.2f}GB of activations without compression"
    )

    store.save(model_name, dataset_path, layers, activations, inputs)


if __name__ == "__main__":
    typer.run(main)

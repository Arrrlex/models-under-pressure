import os
from pathlib import Path

import dotenv
import numpy as np
import typer

from models_under_pressure.config import RESULTS_DIR, GenerateActivationsConfig
from models_under_pressure.dataset.loaders import loaders
from models_under_pressure.probes.model import LLMModel

dotenv.load_dotenv()


def main(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file"),
    model_name: str = typer.Option(
        "meta-llama/Llama-3.2-1B-Instruct", help="Name of model to use"
    ),
    output_dir: Path = typer.Option(
        RESULTS_DIR / "activations", help="Directory to save activations (optional)"
    ),
    loader: str = typer.Option(
        "generated", help="Dataset loader to use (generated, anthropic, toolace)"
    ),  # TODO: it would be nice to only specify one of dataset and loader, and the other is inferred from config somehow
):
    """Generate and save model activations for a dataset."""
    try:
        loader_function = loaders[loader]
    except KeyError:
        available_loaders = ", ".join(loaders.keys())
        raise typer.BadParameter(
            f"Invalid loader '{loader}'. Available loaders: {available_loaders}"
        ) from None

    config = GenerateActivationsConfig(
        dataset_path=dataset_path, model_name=model_name, output_dir=output_dir
    )

    print(config)

    print("Loading model...")
    model = LLMModel.load(
        config.model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )

    print(f"Loading dataset at {config.dataset_path}...")
    dataset = loader_function(config.dataset_path)

    print("Generating activations...")
    activations = model.get_activations(inputs=dataset.inputs)

    print("Saving activations...")
    # Create output directory if it doesn't exist
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save activations array to file
    np.save(config.output_file, activations)


if __name__ == "__main__":
    typer.run(main)

import os

import dotenv
import numpy as np

from models_under_pressure.config import (
    ANTHROPIC_SAMPLES_CSV,
    GenerateActivationsConfig,
)
from models_under_pressure.dataset.loaders import load_anthropic_csv
from models_under_pressure.interfaces.dataset import Dataset, Label
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe, compute_accuracy

dotenv.load_dotenv()


def get_activations(
    model: LLMModel,
    config: GenerateActivationsConfig,
    dataset: Dataset,
    layer: int,
    force_recompute: bool = False,
) -> np.ndarray:
    if config.output_file.exists() and not force_recompute:
        return np.load(config.output_file)["activations"][layer]
    else:
        print("Generating activations...")
        activations = model.get_activations(inputs=dataset.inputs)
        np.savez_compressed(config.output_file, activations=activations)
        return activations[layer]


def test_get_activations(
    layer: int, config: GenerateActivationsConfig, model_name: str, dataset: Dataset
):
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )
    activations = get_activations(
        model=model, config=config, layer=layer, dataset=dataset, force_recompute=True
    )
    # Load precomputed activations
    activations2 = get_activations(
        model=model,
        config=config,
        layer=layer,
        dataset=dataset,
        force_recompute=False,
    )
    assert np.allclose(activations, activations2)


def test_compute_accuracy(
    model_name: str,
    config: GenerateActivationsConfig,
    dataset: Dataset,
    layer: int,
):
    print("Loading model...")
    model = LLMModel.load(
        model_name,
        model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
        tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    )

    print("Loading activations...")
    activations = get_activations(
        model=model,
        config=config,
        layer=layer,
        dataset=dataset,
    )

    if any(label == Label.AMBIGUOUS for label in dataset.labels):
        raise ValueError("Dataset contains ambiguous labels")

    probe = LinearProbe(_llm=model, layer=layer)

    print("Training probe...")
    probe.fit(X=activations, y=dataset.labels_numpy())

    print("Computing accuracy...")
    accuracy = compute_accuracy(probe, dataset, activations)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    dataset_path = ANTHROPIC_SAMPLES_CSV
    layer = 10
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    config = GenerateActivationsConfig(dataset_path=dataset_path, model_name=model_name)
    print("Loading dataset...")
    dataset = load_anthropic_csv(config.dataset_path)

    test_get_activations(
        layer=layer,
        config=config,
        model_name=model_name,
        dataset=dataset,
    )
    # test_compute_accuracy(
    #     model_name=model_name,
    #     config=config,
    #     dataset=dataset,
    #     layer=layer,
    # )

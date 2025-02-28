import os

import dotenv
import numpy as np

from models_under_pressure.config import (
    ANTHROPIC_SAMPLES_CSV,
    GenerateActivationsConfig,
)
from models_under_pressure.dataset.loaders import load_anthropic_csv
from models_under_pressure.interfaces.dataset import Label
from models_under_pressure.probes.model import LLMModel
from models_under_pressure.probes.probes import LinearProbe_, compute_accuracy_

dotenv.load_dotenv()

dataset_path = ANTHROPIC_SAMPLES_CSV
layer = 12


def get_activations(
    model: LLMModel, config: GenerateActivationsConfig, layer: int
) -> np.ndarray:
    if config.output_file.exists():
        return np.load(config.output_file)[layer]
    else:
        print("Generating activations...")
        activations = model.get_activations(inputs=dataset.inputs)
        np.save(config.output_file, activations)
        return activations[layer]


print("Loading model...")
model = LLMModel.load(
    "meta-llama/LLama-3.2-1B-Instruct",
    model_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
    tokenizer_kwargs={"token": os.getenv("HUGGINGFACE_TOKEN")},
)


print("Loading dataset...")
dataset = load_anthropic_csv(dataset_path)

print("Loading activations...")
activations = get_activations(
    GenerateActivationsConfig(dataset_path=dataset_path, model_name=model.name),
    layer=layer,
)

if any(label == Label.AMBIGUOUS for label in dataset.labels):
    raise ValueError("Dataset contains ambiguous labels")

probe = LinearProbe_(_llm=model, layer=layer)

print("Training probe...")
probe.fit(X=activations[layer], y=dataset.labels_numpy())

print("Computing accuracy...")
accuracy = compute_accuracy_(probe, dataset)
print(f"Accuracy: {accuracy}")

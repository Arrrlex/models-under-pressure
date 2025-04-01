from contextlib import contextmanager
import datetime
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel
from safetensors.torch import load_file, save_file
import torch
from tqdm import tqdm

from models_under_pressure.config import (
    ALL_DATASETS,
    LOCAL_MODELS,
    ACTIVATIONS_DIR,
)
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel
from models_under_pressure.interfaces.activations import Activation


class ManifestRow(BaseModel):
    model_name: str
    dataset_name: str
    layer: int
    timestamp: datetime.datetime
    activations: Path
    inputs: Path

    @classmethod
    def build(cls, model_name: str, dataset_name: str, layer: int) -> Self:
        common_id = hashlib.sha1(str(model_name + dataset_name).encode()).hexdigest()[
            :8
        ]
        return cls(
            model_name=model_name,
            dataset_name=dataset_name,
            layer=layer,
            timestamp=datetime.datetime.now(),
            activations=Path(f"activations/{common_id}_{layer}.safetensors"),
            inputs=Path(f"inputs/{common_id}.safetensors"),
        )


class Manifest(BaseModel):
    rows: list[ManifestRow]


@dataclass
class ActivationStore:
    path: Path

    @contextmanager
    def get_manifest(self):
        manifest_path = self.path / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = Manifest.model_validate_json(f.read())
        yield manifest
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))

    def save(
        self,
        model_name: str,
        dataset_name: str,
        layers: list[int],
        activations: torch.Tensor,
        inputs: dict[str, torch.Tensor],
    ):
        # Save layer-specific masked activations
        for layer_idx, layer in tqdm(
            list(enumerate(layers)), desc="Saving activations..."
        ):
            manifest_row = ManifestRow.build(
                model_name=model_name,
                dataset_name=dataset_name,
                layer=layer,
            )

            save_file(
                {"activations": activations[layer_idx]},
                self.path / manifest_row.activations,
            )

            save_file(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                self.path / manifest_row.inputs,
            )

            with self.get_manifest() as manifest:
                manifest.rows.append(manifest_row)

    def load(self, model_name: str, dataset_name: str, layer: int) -> Activation:
        # Load layer-specific masked activation
        manifest_row = ManifestRow.build(
            model_name=model_name, dataset_name=dataset_name, layer=layer
        )
        activations = load_file(self.path / manifest_row.activations)["activations"]
        inputs = load_file(self.path / manifest_row.inputs)
        return Activation(
            _activations=activations.float().numpy(),
            _attention_mask=inputs["attention_mask"].float().numpy(),
            _input_ids=inputs["input_ids"].float().numpy(),
        )

    def delete(self, model_name: str, dataset_name: str, layer: int):
        # Delete layer-specific file
        manifest_row = ManifestRow.build(
            model_name=model_name, dataset_name=dataset_name, layer=layer
        )
        (self.path / manifest_row.activations).unlink()
        (self.path / manifest_row.inputs).unlink()
        with self.get_manifest() as manifest:
            manifest.rows = [
                row
                for row in manifest.rows
                if (row.model_name, row.dataset_name, row.layer)
                != (model_name, dataset_name, layer)
            ]


def compute_activations_and_save(
    model_name: str,
    dataset_name: str,
    layers: list[int],
    activations_dir: Path,
):
    model = LLMModel(model_name)
    dataset = get_dataset(dataset_name)
    print("Getting activations...")
    activations, inputs = model.get_batched_activations(dataset, layers)
    print(
        "Sizes:",
        activations.shape,
        inputs["input_ids"].shape,
        inputs["attention_mask"].shape,
    )
    print(
        "Dtypes:",
        activations.dtype,
        inputs["input_ids"].dtype,
        inputs["attention_mask"].dtype,
    )
    store = ActivationStore(activations_dir)
    store.save(model.name, dataset_name, layers, activations, inputs)


def get_dataset(dataset_name: str) -> LabelledDataset:
    full_dataset_path = ALL_DATASETS[dataset_name]
    return LabelledDataset.load_from(full_dataset_path)


if __name__ == "__main__":
    model_name = LOCAL_MODELS["llama-1b"]
    dataset_name = "synthetic_25_03_25"
    layers = [5, 6, 7, 8]
    activations_dir = ACTIVATIONS_DIR
    # compute_activations_and_save(
    #     model_name=model_name,
    #     dataset_name=dataset_name,
    #     layers=layers,
    #     activations_dir=activations_dir,
    # )

    print("loading activations...")

    activations = ActivationStore(activations_dir).load(
        model_name=model_name,
        dataset_name=dataset_name,
        layer=5,
    )

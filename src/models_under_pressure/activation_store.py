import datetime
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from models_under_pressure.config import (
    ACTIVATIONS_DIR,
    ALL_DATASETS,
    LOCAL_MODELS,
)
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import DatasetSpec, LabelledDataset
from models_under_pressure.model import LLMModel


class ManifestRow(BaseModel):
    model_name: str
    dataset_spec: DatasetSpec
    layer: int
    timestamp: datetime.datetime
    activations: Path
    inputs: Path

    @classmethod
    def build(cls, model_name: str, dataset_spec: DatasetSpec, layer: int) -> Self:
        common_name = model_name + str(dataset_spec.dataset_path)
        common_id = hashlib.sha1(common_name.encode()).hexdigest()[:8]

        return cls(
            model_name=model_name,
            dataset_spec=dataset_spec,
            layer=layer,
            timestamp=datetime.datetime.now(),
            activations=Path(f"activations/{common_id}_{layer}.safetensors"),
            inputs=Path(f"inputs/{common_id}.safetensors"),
        )


class Manifest(BaseModel):
    rows: list[ManifestRow]


@dataclass
class ActivationStore:
    path: Path = ACTIVATIONS_DIR

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
        dataset_spec: DatasetSpec,
        layers: list[int],
        activations: torch.Tensor,
        inputs: dict[str, torch.Tensor],
    ):
        if dataset_spec.indices != "all":
            raise ValueError("Cannot save activations for a subset of the dataset")

        # Save layer-specific masked activations
        for layer_idx, layer in tqdm(
            list(enumerate(layers)), desc="Saving activations..."
        ):
            manifest_row = ManifestRow.build(
                model_name=model_name,
                dataset_spec=dataset_spec,
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

    def load(
        self, model_name: str, dataset_spec: DatasetSpec, layer: int
    ) -> Activation:
        spec_all_indices = dataset_spec.model_copy(update={"indices": "all"})
        # Load layer-specific masked activation
        manifest_row = ManifestRow.build(
            model_name=model_name, dataset_spec=spec_all_indices, layer=layer
        )
        activations = load_file(self.path / manifest_row.activations)["activations"]
        inputs = load_file(self.path / manifest_row.inputs)
        attn_mask = inputs["attention_mask"]
        input_ids = inputs["input_ids"]

        if dataset_spec.indices != "all":
            attn_mask = attn_mask[dataset_spec.indices]
            input_ids = input_ids[dataset_spec.indices]
            activations = activations[dataset_spec.indices]

        return Activation(
            _activations=activations.numpy(),
            _attention_mask=attn_mask.numpy(),
            _input_ids=input_ids.numpy(),
        )

    def delete(self, model_name: str, dataset_spec: DatasetSpec, layer: int):
        # Delete layer-specific file
        manifest_row = ManifestRow.build(
            model_name=model_name, dataset_spec=dataset_spec, layer=layer
        )
        (self.path / manifest_row.activations).unlink()
        (self.path / manifest_row.inputs).unlink()
        with self.get_manifest() as manifest:
            manifest.rows = [
                row
                for row in manifest.rows
                if (row.model_name, row.dataset_spec, row.layer)
                != (model_name, dataset_spec, layer)
            ]


def compute_activations_and_save(
    model_name: str,
    dataset_spec: DatasetSpec,
    layers: list[int],
    activations_dir: Path,
):
    model = LLMModel(model_name)
    dataset = LabelledDataset._load_from(dataset_spec)
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
    store.save(model.name, dataset_spec, layers, activations, inputs)


if __name__ == "__main__":
    model_name = LOCAL_MODELS["llama-1b"]
    dataset_spec = DatasetSpec(
        path=ALL_DATASETS["synthetic_25_03_25"],
        indices="all",
    )
    layers = [5, 6, 7, 8]
    activations_dir = ACTIVATIONS_DIR

    compute_activations_and_save(
        model_name=model_name,
        dataset_spec=dataset_spec,
        layers=layers,
        activations_dir=activations_dir,
    )

    print("loading activations...")

    activations = ActivationStore(activations_dir).load(
        model_name=model_name,
        dataset_spec=dataset_spec,
        layer=5,
    )

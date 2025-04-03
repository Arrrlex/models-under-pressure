import datetime
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from pydantic import BaseModel
from tqdm import tqdm

from models_under_pressure.config import ACTIVATIONS_DIR

from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import DatasetSpec, LabelledDataset
from models_under_pressure.model import LLMModel
from models_under_pressure.r2 import (
    ACTIVATIONS_BUCKET,
    download_file,
    file_exists_in_bucket,
    upload_file,
)


class ManifestRow(BaseModel):
    model_name: str
    dataset_spec: DatasetSpec
    layer: int
    timestamp: datetime.datetime
    activations: Path
    input_ids: Path
    attention_mask: Path

    @classmethod
    def build(cls, model_name: str, dataset_spec: DatasetSpec, layer: int) -> Self:
        common_name = model_name + str(dataset_spec.path)
        common_id = hashlib.sha1(common_name.encode()).hexdigest()[:8]

        return cls(
            model_name=model_name,
            dataset_spec=dataset_spec,
            layer=layer,
            timestamp=datetime.datetime.now(),
            activations=Path(f"activations/{common_id}_{layer}.pt"),
            input_ids=Path(f"input_ids/{common_id}.pt"),
            attention_mask=Path(f"attention_mask/{common_id}.pt"),
        )

    @property
    def paths(self) -> list[Path]:
        return [self.activations, self.input_ids, self.attention_mask]


class Manifest(BaseModel):
    rows: list[ManifestRow]


@dataclass
class ActivationStore:
    path: Path = ACTIVATIONS_DIR
    bucket: str = ACTIVATIONS_BUCKET  # type: ignore

    @contextmanager
    def get_manifest(self):
        manifest_path = self.path / "manifest.json"
        download_file(self.bucket, "manifest.json", manifest_path)
        with open(manifest_path, "r") as f:
            manifest = Manifest.model_validate_json(f.read())
        yield manifest
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))
        upload_file(self.bucket, "manifest.json", manifest_path)

        for row in manifest.rows:
            for path in row.paths:
                if not file_exists_in_bucket(self.bucket, str(path)):
                    upload_file(self.bucket, str(path), self.path / path)

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

            # Save activations
            torch.save(
                activations[layer_idx],
                self.path / manifest_row.activations,
                _use_new_zipfile_serialization=True,
            )

            # Save input_ids
            torch.save(
                inputs["input_ids"],
                self.path / manifest_row.input_ids,
                _use_new_zipfile_serialization=True,
            )

            # Save attention_mask
            torch.save(
                inputs["attention_mask"],
                self.path / manifest_row.attention_mask,
                _use_new_zipfile_serialization=True,
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

        for path in manifest_row.paths:
            key = str(path)
            local_path = self.path / path
            if not local_path.exists():
                download_file(self.bucket, key, local_path)

        # Load activations
        activations = torch.load(self.path / manifest_row.activations)
        if dataset_spec.indices != "all":
            activations = activations[dataset_spec.indices]

        # Load input_ids
        input_ids = torch.load(self.path / manifest_row.input_ids)
        if dataset_spec.indices != "all":
            input_ids = input_ids[dataset_spec.indices]

        # Load attention_mask
        attn_mask = torch.load(self.path / manifest_row.attention_mask)
        if dataset_spec.indices != "all":
            attn_mask = attn_mask[dataset_spec.indices]

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
        (self.path / manifest_row.input_ids).unlink()
        (self.path / manifest_row.attention_mask).unlink()
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
    dataset = LabelledDataset.load_from(dataset_spec)
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

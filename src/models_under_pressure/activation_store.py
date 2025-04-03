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
from models_under_pressure.interfaces.dataset import DatasetSpec
from models_under_pressure.r2 import (
    ACTIVATIONS_BUCKET,
    download_file,
    file_exists_in_bucket,
    upload_file,
)


class ManifestRow(BaseModel):
    model_name: str
    dataset_path: Path
    layer: int
    timestamp: datetime.datetime
    activations: Path
    input_ids: Path
    attention_mask: Path

    @classmethod
    def build(cls, model_name: str, dataset_path: Path, layer: int) -> Self:
        common_name = model_name + str(dataset_path)
        common_id = hashlib.sha1(common_name.encode()).hexdigest()[:8]

        return cls(
            model_name=model_name,
            dataset_path=dataset_path,
            layer=layer,
            timestamp=datetime.datetime.now(),
            activations=Path(f"activations/{common_id}_{layer}.pt"),
            input_ids=Path(f"input_ids/{common_id}.pt"),
            attention_mask=Path(f"attention_masks/{common_id}.pt"),
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
        # Download manifest from R2 and load it
        manifest_path = self.path / "manifest.json"
        download_file(self.bucket, "manifest.json", manifest_path)
        with open(manifest_path, "r") as f:
            manifest = Manifest.model_validate_json(f.read())

        yield manifest

        # Upload modified manifest to R2
        with open(manifest_path, "w") as f:
            f.write(manifest.model_dump_json(indent=2))
        upload_file(self.bucket, "manifest.json", manifest_path)

        # Upload any new files mentioned in manifest to R2
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
                dataset_path=dataset_spec.path,
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
        # Load layer-specific masked activation
        manifest_row = ManifestRow.build(
            model_name=model_name, dataset_path=dataset_spec.path, layer=layer
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

    def delete(self, model_name: str, dataset_path: Path, layer: int):
        # Delete layer-specific file
        manifest_row = ManifestRow.build(
            model_name=model_name, dataset_path=dataset_path, layer=layer
        )

        for path in manifest_row.paths:
            (self.path / path).unlink()

        with self.get_manifest() as manifest:
            manifest.rows = [
                row
                for row in manifest.rows
                if (row.model_name, row.dataset_path, row.layer)
                != (model_name, dataset_path, layer)
            ]

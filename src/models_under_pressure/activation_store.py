import datetime
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self
import tempfile
import os

import torch
from pydantic import BaseModel
from tqdm import tqdm
import zstandard as zstd

from models_under_pressure.config import ACTIVATIONS_DIR, PROJECT_ROOT

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
        dataset_path = dataset_path.resolve().relative_to(PROJECT_ROOT)
        common_name = model_name + str(dataset_path)
        common_id = hashlib.sha1(common_name.encode()).hexdigest()[:8]

        return cls(
            model_name=model_name,
            dataset_path=dataset_path,
            layer=layer,
            timestamp=datetime.datetime.now(),
            activations=Path(f"activations/{common_id}_{layer}.pt.zst"),
            input_ids=Path(f"input_ids/{common_id}.pt.zst"),
            attention_mask=Path(f"attention_masks/{common_id}.pt.zst"),
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

            # Save and compress each tensor
            save_compressed(
                self.path / manifest_row.activations, activations[layer_idx]
            )
            save_compressed(self.path / manifest_row.input_ids, inputs["input_ids"])
            save_compressed(
                self.path / manifest_row.attention_mask, inputs["attention_mask"]
            )

            # Print sizes of saved files
            print(
                f"Activations size: {(self.path / manifest_row.activations).stat().st_size / 1e9:.2f}GB"
            )
            print(
                f"Input IDs size: {(self.path / manifest_row.input_ids).stat().st_size / 1e9:.2f}GB"
            )
            print(
                f"Attention mask size: {(self.path / manifest_row.attention_mask).stat().st_size / 1e9:.2f}GB"
            )

            with self.get_manifest() as manifest:
                manifest.rows.append(manifest_row)

    def load(
        self, model_name: str, dataset_spec: DatasetSpec, layer: int
    ) -> Activation:
        manifest_row = ManifestRow.build(
            model_name=model_name, dataset_path=dataset_spec.path, layer=layer
        )

        if not self.exists(model_name, dataset_spec.path, layer):
            raise FileNotFoundError(
                f"Activations for {model_name} on {dataset_spec.path} at layer {layer} not found"
            )

        for path in manifest_row.paths:
            key = str(path)
            local_path = self.path / path
            if not local_path.exists():
                download_file(self.bucket, key, local_path)

        # Load and decompress each file
        activations = load_compressed(self.path / manifest_row.activations)
        input_ids = load_compressed(self.path / manifest_row.input_ids)
        attn_mask = load_compressed(self.path / manifest_row.attention_mask)

        if dataset_spec.indices != "all":
            activations = activations[dataset_spec.indices]
            input_ids = input_ids[dataset_spec.indices]
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
                if row.activations != manifest_row.activations
            ]

    def exists(self, model_name: str, dataset_path: Path, layer: int) -> bool:
        row = ManifestRow.build(
            model_name=model_name, dataset_path=dataset_path, layer=layer
        )
        with self.get_manifest() as manifest:
            return any(row.activations == other.activations for other in manifest.rows)


@contextmanager
def temp_file():
    """Create a temporary file that is deleted after use."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        try:
            yield Path(tmp.name)
        finally:
            os.unlink(tmp.name)


def load_compressed(path: Path) -> torch.Tensor:
    dctx = zstd.ZstdDecompressor()
    with temp_file() as tmp_path:
        with open(path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            dctx.copy_stream(f_in, f_out)

        return torch.load(tmp_path).cpu()


def save_compressed(path: Path, tensor: torch.Tensor):
    with temp_file() as tmp_path:
        # Save tensor to temporary file
        torch.save(tensor, tmp_path)

        # Compress with zstd
        cctx = zstd.ZstdCompressor(level=10)
        with open(tmp_path, "rb") as f_in, open(path, "wb") as f_out:
            cctx.copy_stream(f_in, f_out)

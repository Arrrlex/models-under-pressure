"""Module for storing and managing model activations.

This module provides functionality for storing, loading, and managing model activations
in a compressed format. It handles the persistence of activations to both local storage
and cloud storage (R2), with a manifest system to track available activations.
"""

import datetime
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import torch
import zstandard as zstd
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from models_under_pressure.config import PROJECT_ROOT, global_settings
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.r2 import (
    ACTIVATIONS_BUCKET,
    delete_file,
    download_file,
    list_bucket_files,
    upload_file,
)


class ActivationsSpec(BaseModel):
    """Specification for a set of activations.

    This class represents a set of activations, including the model name,
    dataset path, and layer number.
    """

    model_name: str
    dataset_path: Path
    layer: int

    class Config:
        frozen = True  # Make the model immutable
        validate_assignment = True

    @field_validator("dataset_path")
    def validate_dataset_path(cls, v: Path) -> Path:
        return v.resolve().relative_to(PROJECT_ROOT)


class ManifestRow(BaseModel):
    """Represents a single row in the activation manifest.

    This class tracks metadata about stored activations, including the model name,
    dataset path, layer number, and paths to the stored activation files.
    """

    model_name: str
    dataset_path: Path
    layer: int
    timestamp: datetime.datetime
    activations: Path
    input_ids: Path
    attention_mask: Path

    @classmethod
    def from_spec(cls, spec: ActivationsSpec) -> Self:
        """Create a new manifest row with generated file paths.

        Args:
            model_name: Name of the model that generated the activations
            dataset_path: Path to the dataset used
            layer: Layer number for which activations were generated

        Returns:
            A new ManifestRow instance with generated file paths
        """
        common_name = spec.model_name + str(spec.dataset_path)
        common_id = hashlib.sha1(common_name.encode()).hexdigest()[:8]

        return cls(
            model_name=spec.model_name,
            dataset_path=spec.dataset_path,
            layer=spec.layer,
            timestamp=datetime.datetime.now(),
            activations=Path(f"activations/{common_id}_{spec.layer}.pt.zst"),
            input_ids=Path(f"input_ids/{common_id}.pt.zst"),
            attention_mask=Path(f"attention_masks/{common_id}.pt.zst"),
        )

    @property
    def paths(self) -> list[Path]:
        """Get all file paths associated with this manifest row.

        Returns:
            List of paths to activation files
        """
        return [self.activations, self.input_ids, self.attention_mask]

    @property
    def spec(self) -> ActivationsSpec:
        """Get the specification for this manifest row.

        Returns:
            The specification for this manifest row
        """
        return ActivationsSpec(
            model_name=self.model_name, dataset_path=self.dataset_path, layer=self.layer
        )


class Manifest(BaseModel):
    """Container for all manifest rows.

    This class represents the complete manifest of stored activations,
    containing a list of all ManifestRow instances.
    """

    rows: list[ManifestRow]

    def to_lookup(self) -> dict[ActivationsSpec, ManifestRow]:
        """Lookup a manifest row by its specification.

        Returns:
            A dictionary mapping specifications to manifest rows
        """
        return {row.spec: row for row in self.rows}

    def add_rows(self, rows: list[ManifestRow]):
        lookup = self.to_lookup()
        lookup.update({row.spec: row for row in rows})
        self.rows = [lookup[spec] for spec in lookup]

    def remove_rows(self, rows: list[ManifestRow]):
        lookup = self.to_lookup()
        for row in rows:
            lookup.pop(row.spec)
        self.rows = [lookup[spec] for spec in lookup]


@dataclass
class ActivationStore:
    """Manages storage and retrieval of model activations.

    This class handles the persistence of model activations, including:
    - Saving activations to local storage and cloud storage
    - Loading activations from storage
    - Managing the manifest of available activations
    - Deleting stored activations
    - Checking for existence of activations

    Attributes:
        path: Local directory path for storing activations
        bucket: Cloud storage bucket name for storing activations
    """

    path: Path = global_settings.ACTIVATIONS_DIR
    bucket: str = ACTIVATIONS_BUCKET  # type: ignore
    manifest: Manifest = field(init=False)

    def __post_init__(self):
        self.manifest = self.load_manifest()

    def load_manifest(self) -> Manifest:
        download_file(self.bucket, "manifest.json", self.manifest_path)
        with open(self.manifest_path, "r") as f:
            return Manifest.model_validate_json(f.read())

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def save_manifest(self):
        with open(self.manifest_path, "w") as f:
            f.write(self.manifest.model_dump_json(indent=2))

        upload_file(self.bucket, "manifest.json", self.manifest_path)

    def sync(
        self, add_specs: list[ActivationsSpec], remove_specs: list[ActivationsSpec]
    ):
        self.manifest = self.load_manifest()
        self.manifest.add_rows([ManifestRow.from_spec(spec) for spec in add_specs])
        self.manifest.remove_rows(
            [ManifestRow.from_spec(spec) for spec in remove_specs]
        )
        self.save_manifest()

        manifest_paths = {str(path) for row in self.manifest.rows for path in row.paths}
        local_paths = {
            str(path.relative_to(self.path)) for path in self.path.glob("**/*.pt.zst")
        }
        remote_paths = {
            path for path in list_bucket_files(self.bucket) if path.endswith(".pt.zst")
        }

        # Delete local files that are not in the manifest
        for path in local_paths - manifest_paths:
            print(f"Deleting {path} from local")
            (self.path / path).unlink()
            (self.path / path).with_suffix("").unlink()

        # Delete remote files that are not in the manifest
        for path in remote_paths - manifest_paths:
            print(f"Deleting {path} from remote")
            delete_file(self.bucket, path)

        # Any remaining local files that aren't in remote are new activations, upload them
        for path in local_paths - remote_paths:
            print(f"Uploading {path} to remote")
            upload_file(self.bucket, path, self.path / path)

        # Delete from the manifest any activations that are not present locally or remotely
        for path in manifest_paths - local_paths - remote_paths:
            print(f"Removing {path} from manifest")
            self.manifest.rows = [
                row for row in self.manifest.rows if row.activations != path
            ]

        self.save_manifest()

    def save(
        self,
        model_name: str,
        dataset_path: Path,
        layers: list[int],
        activations: torch.Tensor,
        inputs: dict[str, torch.Tensor],
    ):
        """Save model activations to storage.

        Args:
            model_name: Name of the model that generated the activations
            dataset_path: Path to the dataset used
            layers: List of layer numbers for which activations were generated
            activations: Tensor containing the model activations
            inputs: Dictionary containing input tensors (input_ids and attention_mask)

        Raises:
            ValueError: If trying to save activations for a subset of the dataset
        """
        # Save layer-specific masked activations
        for layer_idx, layer in tqdm(
            list(enumerate(layers)), desc="Saving activations..."
        ):
            spec = ActivationsSpec(
                model_name=model_name,
                dataset_path=dataset_path,
                layer=layer,
            )
            manifest_row = ManifestRow.from_spec(spec)

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

            self.sync(add_specs=[spec], remove_specs=[])

    def load(
        self, spec: ActivationsSpec
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load stored activations from storage.

        Args:
            spec: Specification for the activations to load

        Returns:
            An Activation object containing the loaded activations and inputs

        Raises:
            FileNotFoundError: If the requested activations are not found in storage
        """
        manifest_row = ManifestRow.from_spec(spec)

        if not self.exists(spec):
            raise FileNotFoundError(f"Activations for {spec} not found")

        for path in manifest_row.paths:
            key = str(path)
            local_path = self.path / path
            if not local_path.exists():
                download_file(self.bucket, key, local_path)

        # Load and decompress each file
        activations = load_compressed(self.path / manifest_row.activations)
        input_ids = load_compressed(self.path / manifest_row.input_ids)
        attn_mask = load_compressed(self.path / manifest_row.attention_mask)

        return activations, input_ids, attn_mask

    def exists(self, spec: ActivationsSpec) -> bool:
        """Check if activations exist in storage.

        Args:
            spec: Specification for the activations to check

        Returns:
            True if the activations exist, False otherwise
        """
        return spec in self.manifest.to_lookup()

    def enrich(
        self, dataset: LabelledDataset, path: Path, model_name: str, layer: int
    ) -> LabelledDataset:
        spec = ActivationsSpec(model_name=model_name, dataset_path=path, layer=layer)
        activations, input_ids, attn_mask = self.load(spec)
        return dataset.assign(
            activations=activations,
            attention_mask=attn_mask,
            input_ids=input_ids,
        )


def load_compressed(path: Path) -> torch.Tensor:
    """Load and decompress a tensor from a compressed file.

    Args:
        path: Path to the compressed tensor file

    Returns:
        The decompressed tensor
    """
    dctx = zstd.ZstdDecompressor()
    tmp_path = path.with_suffix("")
    if not tmp_path.exists():
        file_size = os.path.getsize(path)
        with open(path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            with tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Decompressing"
            ) as pbar:
                for chunk in dctx.read_to_iter(f_in, write_size=16 * 1024 * 1024):
                    f_out.write(chunk)
                    pbar.update(f_in.tell() - pbar.n)

    return torch.load(tmp_path, map_location="cpu", mmap=True)


def save_compressed(path: Path, tensor: torch.Tensor):
    """Save and compress a tensor to a file.

    Args:
        path: Path where to save the compressed tensor
        tensor: The tensor to compress and save
    """
    if not path.name.endswith(".pt.zst"):
        raise ValueError("Path must have .pt.zst suffix")
    tmp_path = path.with_suffix("")

    start = time.time()
    torch.save(tensor, tmp_path)
    end = time.time()
    print(f"Saved tensor in {end - start:.2f} seconds")

    # Compress with zstd
    cctx = zstd.ZstdCompressor(level=4)
    file_size = os.path.getsize(tmp_path)
    with open(tmp_path, "rb") as f_in, open(path, "wb") as f_out:
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Compressing"
        ) as pbar:
            for chunk in cctx.read_to_iter(f_in, write_size=16 * 1024 * 1024):
                f_out.write(chunk)
                pbar.update(f_in.tell() - pbar.n)

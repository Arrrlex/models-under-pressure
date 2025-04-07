"""Module for storing and managing model activations.

This module provides functionality for storing, loading, and managing model activations
in a compressed format. It handles the persistence of activations to both local storage
and cloud storage (R2), with a manifest system to track available activations.
"""

import datetime
import hashlib
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
import zstandard as zstd
from pydantic import BaseModel
from tqdm import tqdm

from models_under_pressure.config import ACTIVATIONS_DIR, PROJECT_ROOT
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.r2 import (
    ACTIVATIONS_BUCKET,
    download_file,
    file_exists_in_bucket,
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
        dataset_path = spec.dataset_path.resolve().relative_to(PROJECT_ROOT)
        common_name = spec.model_name + str(dataset_path)
        common_id = hashlib.sha1(common_name.encode()).hexdigest()[:8]

        return cls(
            model_name=spec.model_name,
            dataset_path=dataset_path,
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


class Manifest(BaseModel):
    """Container for all manifest rows.

    This class represents the complete manifest of stored activations,
    containing a list of all ManifestRow instances.
    """

    rows: list[ManifestRow]


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

    path: Path = ACTIVATIONS_DIR
    bucket: str = ACTIVATIONS_BUCKET  # type: ignore

    @contextmanager
    def get_manifest(self):
        """Context manager for accessing and updating the manifest.

        Downloads the manifest from cloud storage, yields it for modification,
        and then uploads the modified manifest back to cloud storage.

        Yields:
            The current manifest object
        """
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

            with self.get_manifest() as manifest:
                manifest.rows.append(manifest_row)

    def load(self, spec: ActivationsSpec) -> Activation:
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

        return Activation(
            _activations=activations.numpy(),
            _attention_mask=attn_mask.numpy(),
            _input_ids=input_ids.numpy(),
        )

    def delete(self, spec: ActivationsSpec):
        """Delete stored activations from storage.

        Args:
            model_name: Name of the model that generated the activations
        """
        # Delete layer-specific file
        manifest_row = ManifestRow.from_spec(spec)

        for path in manifest_row.paths:
            (self.path / path).unlink()

        with self.get_manifest() as manifest:
            manifest.rows = [
                row
                for row in manifest.rows
                if row.activations != manifest_row.activations
            ]

    def exists(self, spec: ActivationsSpec) -> bool:
        """Check if activations exist in storage.

        Args:
            spec: Specification for the activations to check

        Returns:
            True if the activations exist, False otherwise
        """
        row = ManifestRow.from_spec(spec)
        with self.get_manifest() as manifest:
            return any(row.activations == other.activations for other in manifest.rows)

    def enrich(
        self, dataset: LabelledDataset, spec: ActivationsSpec
    ) -> LabelledDataset:
        """Enrich a dataset with activations.

        Args:
            dataset: The dataset to enrich
            spec: Specification for the activations to load

        Returns:
            The enriched dataset
        """
        activations = self.load(spec)
        return dataset.assign(
            activations=activations.get_activations(),
            attention_mask=activations.get_attention_mask(),
            input_ids=activations.get_input_ids(),
        )


@contextmanager
def temp_file():
    """Create a temporary file that is deleted after use.

    Yields:
        Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        try:
            yield Path(tmp.name)
        finally:
            os.unlink(tmp.name)


def load_compressed(path: Path) -> torch.Tensor:
    """Load and decompress a tensor from a compressed file.

    Args:
        path: Path to the compressed tensor file

    Returns:
        The decompressed tensor
    """
    dctx = zstd.ZstdDecompressor()
    with temp_file() as tmp_path:
        with open(path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            dctx.copy_stream(f_in, f_out)

        return torch.load(tmp_path).cpu()


def save_compressed(path: Path, tensor: torch.Tensor):
    """Save and compress a tensor to a file.

    Args:
        path: Path where to save the compressed tensor
        tensor: The tensor to compress and save
    """
    with temp_file() as tmp_path:
        # Save tensor to temporary file
        torch.save(tensor, tmp_path)

        # Compress with zstd
        cctx = zstd.ZstdCompressor(level=10)
        with open(tmp_path, "rb") as f_in, open(path, "wb") as f_out:
            # Get file size for progress bar
            file_size = os.path.getsize(tmp_path)
            with tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Compressing"
            ) as pbar:
                while True:
                    chunk = f_in.read(8192)  # Read in 8KB chunks
                    if not chunk:
                        break
                    compressed = cctx.compress(chunk)
                    f_out.write(compressed)
                    pbar.update(len(chunk))

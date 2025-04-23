from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import hashlib
import json
from pathlib import Path
import pickle
from typing import Self

from pydantic import BaseModel

from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.probes.base import Probe
from models_under_pressure.config import PROBES_DIR


class ProbeRegistry(BaseModel):
    probes: list["ProbeManifest"]

    @classmethod
    @contextmanager
    def open(cls, path: Path):
        with open(path, "r") as f:
            registry = cls.model_validate_json(f.read())
        yield registry
        with open(path, "w") as f:
            f.write(registry.model_dump_json(indent=2))


class ProbeManifest(BaseModel):
    spec: ProbeSpec
    model_name: str
    layer: int
    train_dataset_hash: str
    validation_dataset_hash: str | None
    timestamp: datetime.datetime
    probe_id: str

    @classmethod
    def from_spec(
        cls,
        spec: ProbeSpec,
        model_name: str,
        layer: int,
        train_dataset_hash: str,
        validation_dataset_hash: str | None,
    ) -> Self:
        probe_info = json.dumps(
            {
                "spec": spec.model_dump(),
                "model_name": model_name,
                "layer": layer,
                "train_dataset_hash": train_dataset_hash,
                "validation_dataset_hash": validation_dataset_hash,
            }
        )
        probe_id = hashlib.sha256(probe_info.encode()).hexdigest()[:8]
        return cls(
            spec=spec,
            model_name=model_name,
            layer=layer,
            train_dataset_hash=train_dataset_hash,
            validation_dataset_hash=validation_dataset_hash,
            timestamp=datetime.datetime.now(),
            probe_id=probe_id,
        )


@dataclass
class ProbeStore:
    path: Path = PROBES_DIR

    def __post_init__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            with open(self.registry_path, "w") as f:
                f.write(ProbeRegistry(probes=[]).model_dump_json())

    @property
    def registry_path(self) -> Path:
        return self.path / "registry.json"

    def save(
        self,
        probe: Probe,
        spec: ProbeSpec,
        model_name: str,
        layer: int,
        train_dataset_hash: str,
        validation_dataset_hash: str | None,
    ):
        manifest = ProbeManifest.from_spec(
            spec, model_name, layer, train_dataset_hash, validation_dataset_hash
        )
        probe_path = self.path / f"{manifest.probe_id}.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)

        with ProbeRegistry.open(self.registry_path) as registry:
            registry.probes.append(manifest)

    def load(
        self,
        spec: ProbeSpec,
        model_name: str,
        layer: int,
        train_dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None,
    ) -> Probe:
        train_dataset_hash = train_dataset.hash
        if validation_dataset is not None:
            validation_dataset_hash = validation_dataset.hash
        else:
            validation_dataset_hash = None
        manifest = ProbeManifest.from_spec(
            spec, model_name, layer, train_dataset_hash, validation_dataset_hash
        )
        probe_path = self.path / f"{manifest.probe_id}.pkl"
        with open(probe_path, "rb") as f:
            return pickle.load(f)

    def delete(
        self,
        spec: ProbeSpec,
        model_name: str,
        layer: int,
        train_dataset_hash: str,
        validation_dataset_hash: str | None,
    ):
        manifest = ProbeManifest.from_spec(
            spec, model_name, layer, train_dataset_hash, validation_dataset_hash
        )
        probe_path = self.path / f"{manifest.probe_id}.pkl"
        if probe_path.exists():
            probe_path.unlink()
        with ProbeRegistry.open(self.registry_path) as registry:
            registry.probes = [
                p for p in registry.probes if p.probe_id != manifest.probe_id
            ]

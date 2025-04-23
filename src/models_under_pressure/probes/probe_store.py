from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import hashlib
from pathlib import Path
import pickle
from typing import Self

from pydantic import BaseModel

from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.probes.base import Probe
from models_under_pressure.config import PROBES_DIR


class FullProbeSpec(ProbeSpec):
    model_name: str
    layer: int
    train_dataset_hash: str
    validation_dataset_hash: str | None

    @classmethod
    def from_spec(
        cls,
        spec: ProbeSpec,
        train_dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None,
        model_name: str,
        layer: int,
    ) -> Self:
        return cls(
            **spec.model_dump(),
            train_dataset_hash=train_dataset.hash,
            validation_dataset_hash=validation_dataset.hash
            if validation_dataset
            else None,
            model_name=model_name,
            layer=layer,
        )


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


class ProbeManifest(FullProbeSpec):
    timestamp: datetime.datetime
    probe_hash: str

    @classmethod
    def from_full_spec(cls, spec: FullProbeSpec) -> Self:
        return cls(
            **spec.model_dump(),
            timestamp=datetime.datetime.now(),
            probe_hash=hashlib.sha256(str(spec.model_dump()).encode()).hexdigest()[:8],
        )

    @classmethod
    def from_spec(
        cls,
        spec: ProbeSpec,
        train_dataset: LabelledDataset,
        validation_dataset: LabelledDataset | None,
    ) -> Self:
        full_spec = FullProbeSpec.from_spec(spec, train_dataset, validation_dataset)
        return cls.from_full_spec(full_spec)


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

    def exists(self, spec: FullProbeSpec) -> bool:
        manifest = ProbeManifest.from_full_spec(spec)
        return (self.path / f"{manifest.probe_hash}.pkl").exists()

    def save(
        self,
        probe: Probe,
        spec: FullProbeSpec,
    ):
        manifest = ProbeManifest.from_full_spec(spec)

        with ProbeRegistry.open(self.registry_path) as registry:
            registry.probes.append(manifest)

        probe_path = self.path / f"{manifest.probe_hash}.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)

    def load(self, spec: FullProbeSpec) -> Probe:
        manifest = ProbeManifest.from_full_spec(spec)
        probe_path = self.path / f"{manifest.probe_hash}.pkl"
        with open(probe_path, "rb") as f:
            return pickle.load(f)

    def delete(self, spec: FullProbeSpec):
        manifest = ProbeManifest.from_full_spec(spec)
        probe_path = self.path / f"{manifest.probe_hash}.pkl"
        if probe_path.exists():
            probe_path.unlink()
        with ProbeRegistry.open(self.registry_path) as registry:
            registry.probes = [
                p for p in registry.probes if p.probe_hash != manifest.probe_hash
            ]

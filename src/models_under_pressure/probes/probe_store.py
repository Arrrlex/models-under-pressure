import hashlib
import json
import pickle
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel

from models_under_pressure.config import PROBES_DIR
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.probes import ProbeSpec
from models_under_pressure.probes.base import Probe


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

    @property
    def hash(self) -> str:
        dumped = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(dumped.encode()).hexdigest()[:8]


class Registry(BaseModel):
    probes: dict[str, FullProbeSpec]

    @classmethod
    @contextmanager
    def open(cls, path: Path):
        registry = cls.model_validate_json(path.read_text())
        yield registry
        path.write_text(registry.model_dump_json(indent=2))


@dataclass
class ProbeStore:
    path: Path = PROBES_DIR

    def __post_init__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text(Registry(probes={}).model_dump_json())

    @property
    def registry_path(self) -> Path:
        return self.path / "registry.json"

    def exists(self, spec: FullProbeSpec) -> bool:
        return (self.path / f"{spec.hash}.pkl").exists()

    def save(
        self,
        probe: Probe,
        spec: FullProbeSpec,
    ):
        with Registry.open(self.registry_path) as registry:
            registry.probes[spec.hash] = spec

        probe_path = self.path / f"{spec.hash}.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)

    def load(self, spec: FullProbeSpec) -> Probe:
        return self.load_from_id(spec.hash)

    def load_from_id(self, probe_id: str) -> Probe:
        with open(self.path / f"{probe_id}.pkl", "rb") as f:
            return pickle.load(f)

    def delete(self, spec: FullProbeSpec):
        probe_path = self.path / f"{spec.hash}.pkl"
        if probe_path.exists():
            probe_path.unlink()
        with Registry.open(self.registry_path) as registry:
            del registry.probes[spec.hash]

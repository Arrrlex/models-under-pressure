from contextlib import contextmanager
from dataclasses import dataclass
import datetime
import hashlib
import json
from pathlib import Path
import pickle
from typing import Self

from pydantic import BaseModel
from transformers import AutoTokenizer

from models_under_pressure.interfaces.probes import ProbeSpec, ProbeType
from models_under_pressure.probes.base import Probe
from models_under_pressure.probes import aggregations as agg
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
            f.write(registry.model_dump_json())


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
        spec = prune_spec(spec)
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
        train_dataset_hash: str,
        validation_dataset_hash: str | None,
    ) -> Probe:
        original_spec = spec
        spec = prune_spec(spec)
        manifest = ProbeManifest.from_spec(
            spec, model_name, layer, train_dataset_hash, validation_dataset_hash
        )
        probe_path = self.path / f"{manifest.probe_id}.pkl"
        with open(probe_path, "rb") as f:
            probe = pickle.load(f)
        return add_back_pruned_stuff(probe, original_spec, model_name)

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


def prune_spec(spec: ProbeSpec) -> ProbeSpec:
    if spec.name in [
        ProbeType.per_entry,
        ProbeType.difference_of_means,
        ProbeType.lda,
        ProbeType.attention,
    ]:
        pass
    elif spec.name in [
        ProbeType.max,
        ProbeType.mean,
        ProbeType.last,
        ProbeType.max_of_rolling_mean,
        ProbeType.mean_of_top_k,
        ProbeType.max_of_sentence_means,
    ]:
        spec.name = ProbeType.mean
    else:
        raise ValueError(f"Need to add support for {spec.name} to prune_spec")
    return spec


def add_back_pruned_stuff(
    probe: Probe, original_spec: ProbeSpec, model_name: str
) -> Probe:
    match original_spec.name:
        case ProbeType.max:
            probe._classifier.aggregation_method = agg.Max()
        case ProbeType.mean:
            probe._classifier.aggregation_method = agg.Mean()
        case ProbeType.last:
            probe._classifier.aggregation_method = agg.Last()
        case ProbeType.max_of_rolling_mean:
            window_size = original_spec.hyperparams["window_size"]
            probe._classifier.aggregation_method = agg.MaxOfRollingMean(
                window_size=window_size
            )
        case ProbeType.mean_of_top_k:
            k = original_spec.hyperparams["k"]
            probe._classifier.aggregation_method = agg.MeanOfTopK(k=k)
        case ProbeType.max_of_sentence_means:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            probe._classifier.aggregation_method = agg.MaxOfSentenceMeans(
                tokenizer=tokenizer
            )
        case ProbeType.per_entry:
            pass
        case ProbeType.difference_of_means:
            pass
        case ProbeType.lda:
            pass
        case ProbeType.attention:
            pass
        case _:
            raise ValueError(
                f"Need to add support for {original_spec.name} to add_back_pruned_stuff"
            )
    return probe

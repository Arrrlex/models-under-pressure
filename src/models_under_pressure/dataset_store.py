import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel
from tqdm import tqdm

from models_under_pressure.config import PROJECT_ROOT
from models_under_pressure.r2 import (
    DATASETS_BUCKET,
    download_file,
    file_exists_in_bucket,
    upload_file,
)


class DatasetManifest(BaseModel):
    path: Path
    md5: str

    @classmethod
    def from_path(cls, path: Path) -> Self:
        return cls(path=path, md5=md5(path))


class DatasetRegistry(BaseModel):
    datasets: list[DatasetManifest]

    @property
    def paths(self) -> list[Path]:
        return [dataset.path for dataset in self.datasets]


@dataclass
class DatasetStore:
    bucket: str = DATASETS_BUCKET  # type: ignore

    def load_registry(self) -> DatasetRegistry:
        download_file(self.bucket, "registry.json", self.registry_path)
        with open(self.registry_path, "r") as f:
            return DatasetRegistry.model_validate_json(f.read())

    @property
    def registry_path(self) -> Path:
        return PROJECT_ROOT / "data/registry.json"

    def save_registry(self, registry: DatasetRegistry):
        with open(self.registry_path, "w") as f:
            f.write(registry.model_dump_json(indent=2))

        upload_file(self.bucket, "registry.json", self.registry_path)

    def upload(self, paths: list[Path]):
        for path in tqdm(paths, desc="Uploading datasets"):
            path = path.resolve().relative_to(PROJECT_ROOT)
            registry = self.load_registry()
            if path not in registry.paths:
                if file_exists_in_bucket(self.bucket, str(path)):
                    print(f"Dataset {path} already exists in bucket")
                else:
                    upload_file(self.bucket, str(path), PROJECT_ROOT / path)
                registry.datasets.append(DatasetManifest.from_path(path))
            else:
                print(f"Dataset {path} already exists")
        self.save_registry(registry)

    def download_all(self):
        registry = self.load_registry()
        to_download = []
        for dataset in registry.datasets:
            if (PROJECT_ROOT / dataset.path).exists():
                if md5(PROJECT_ROOT / dataset.path) != dataset.md5:
                    print(f"Warning: Dataset {dataset.path} has changed")
            else:
                to_download.append(dataset)
        if not to_download:
            print("All datasets are already downloaded")
            return
        for dataset in tqdm(to_download, desc="Downloading datasets"):
            download_file(
                bucket_name=self.bucket,
                key=str(dataset.path),
                local_path=PROJECT_ROOT / dataset.path,
            )

    def delete(self, path: Path):
        path = path.resolve().relative_to(PROJECT_ROOT)
        registry = self.load_registry()
        if path not in registry.paths:
            print(f"Dataset {path} not found")
        else:
            registry.datasets = [
                dataset for dataset in registry.datasets if dataset.path != path
            ]
            self.save_registry(registry)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

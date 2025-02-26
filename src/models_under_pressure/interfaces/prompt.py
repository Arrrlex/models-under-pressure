import abc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class Prompt(abc.ABC):
    def __init__(
        self,
        id: int,
        prompt: str,
        situations: Dict[str, int],
        high_stakes: bool,
        timestamp: str,
        topic: str | None = None,
        factors: str | None = None,
        metadata: Dict[str, str] | None = None,
        variation: Optional[Dict[str, str]] = None,
    ):
        self.id = id
        self.prompt = prompt
        assert "high_stakes" in situations
        assert "low_stakes" in situations
        self.situations = situations

        self.high_stakes = high_stakes
        self.timestamp = timestamp

        self.topic = topic
        self.factors = factors
        self.variation = {}

        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        self.metadata.update(metadata)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        prompt_dict = {
            "id": self.id,
            "prompt": self.prompt,
            "situations": self.situations,
            "topic": self.topic,
            "factors": self.factors,
            "high_stakes": self.high_stakes,
            "timestamp": self.timestamp,
            **self.variation,
        }
        if include_metadata:
            prompt_dict["metadata"] = self.metadata
        return prompt_dict

    def to_json(self, include_metadata: bool = False) -> str:
        return json.dumps(self.to_dict(include_metadata=include_metadata))

    def metadata_to_json(self) -> str:
        metadata = self.metadata | {"id": self.id, "prompt": self.prompt}
        return json.dumps(metadata)

    @classmethod
    def to_jsonl(
        cls, prompts: List["Prompt"], file_path: Path, mode: str = "a"
    ) -> None:
        with open(file_path, mode) as f:
            for prompt in prompts:
                f.write(prompt.to_json() + "\n")

    @classmethod
    def metadata_to_jsonl(
        cls, prompts: List["Prompt"], file_path: Path, mode: str = "a"
    ) -> None:
        with open(file_path, mode) as f:
            for prompt in prompts:
                f.write(prompt.metadata_to_json() + "\n")

    @classmethod
    def from_jsonl(
        cls, file_path: Path, metadata_file_path: Path | None = None
    ) -> List["Prompt"]:
        """Load prompts from JSONL file and optionally add metadata from a separate file.

        Args:
            file_path: Path to the prompts JSONL file
            metadata_file_path: Optional path to metadata JSONL file. If provided, metadata will be loaded and merged.
        """
        prompt_dicts = [json.loads(line) for line in open(file_path)]

        if metadata_file_path is None or not metadata_file_path.exists():
            metadata = [{} for _ in prompt_dicts]
        else:
            # Create a dictionary mapping prompt IDs to their metadata
            metadata_dicts = [json.loads(line) for line in open(metadata_file_path)]

            if not len(metadata_dicts):
                metadata = [{} for _ in prompt_dicts]
            else:
                metadata_by_id = {}
                for metadata_dict in metadata_dicts:
                    metadata_id = int(metadata_dict["id"])
                    metadata_by_id[metadata_id] = {
                        k: v
                        for k, v in metadata_dict.items()
                        if k not in ["id", "prompt"]
                    }

                # Match metadata to prompts by ID
                metadata = [metadata_by_id.get(pd["id"], {}) for pd in prompt_dicts]

        return [
            cls(**prompt_dict, metadata=mdata)
            for prompt_dict, mdata in zip(prompt_dicts, metadata)
        ]

from typing import Any, Dict


class Variation:
    """Represents a single Variation entity over which we want to generate a set of prompts.

    This is
    """

    def __init__(
        self,
        id: int,
        name: str,
        description: str,
    ):
        self.id = id
        self.name = name
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }

    def __repr__(self):
        return f"Variation(id={self.id}, description='{self.description}', name='{self.name}')"

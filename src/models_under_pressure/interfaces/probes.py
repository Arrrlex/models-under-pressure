from typing import Any
from pydantic import BaseModel


class ProbeSpec(BaseModel):
    name: str
    hyperparams: dict[str, Any] | None = None

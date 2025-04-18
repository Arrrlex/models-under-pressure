from pydantic import BaseModel
from pydantic import JsonValue


class ProbeSpec(BaseModel):
    name: str
    hyperparams: dict[str, JsonValue]

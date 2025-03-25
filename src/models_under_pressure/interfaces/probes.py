from pydantic import BaseModel


class ProbeSpec(BaseModel):
    name: str
    preprocessor: str | None = None
    postprocessor: str | None = None

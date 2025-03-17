from pydantic import BaseModel


class Variation(BaseModel):
    type: str
    name: str
    value: str

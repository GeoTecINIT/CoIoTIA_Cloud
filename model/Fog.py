from pydantic import BaseModel

class Fog(BaseModel):
    id: int
    name: str
    ip: str
    status: str
    cpu: float | None
    ram: float | None
    disk: float | None


class FogVitals(BaseModel):
    cpu: float | None
    ram: float | None
    disk: float | None
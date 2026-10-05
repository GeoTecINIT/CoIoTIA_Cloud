from pydantic import BaseModel
from typing import Literal

class MultiPolygon(BaseModel):
    type: Literal["MultiPolygon"]
    coordinates: list[list[list[list[float]]]]

class Coi(BaseModel):
    id: int
    geom: MultiPolygon
    silhouette_score: float | None
    use_case_id: int
    fog_id: int | None

class Centroid(BaseModel):
    lat: float
    lon: float
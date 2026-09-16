from pydantic import BaseModel

class Domain(BaseModel):
    id: int
    name: str
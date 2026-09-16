from pydantic import BaseModel

class UseCase(BaseModel):
    name: str
    description: str
    keywords: str
    clustering_method: str
    domain_id: int

class UseCaseCreate(UseCase):
    pass

class UseCaseRead(UseCase):
    id: int
    user_id: str

class UseCaseCreateResponse(BaseModel):
    id: int


from pydantic import BaseModel

class AnalysisType(BaseModel):
    id: int
    name: str

class DataType(BaseModel):
    id: int
    name: str

class TypesResponse(BaseModel):
    analysis_types: list[AnalysisType]
    data_types: list[DataType]
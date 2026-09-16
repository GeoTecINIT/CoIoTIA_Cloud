from fastapi import APIRouter, Request

from model.Type import TypesResponse

router = APIRouter()

@router.get("/getTypes", response_model=TypesResponse)
async def get_domains(request: Request):
    service = request.app.state.types_service
    analysis_types = await service.get_analysis_types()
    data_types = await service.get_data_types()
    return TypesResponse(
        analysis_types=analysis_types,
        data_types=data_types 
    )

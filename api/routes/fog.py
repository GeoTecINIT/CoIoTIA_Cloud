from fastapi import APIRouter, Request

from model.Fog import Fog, FogVitals

router = APIRouter()

@router.get("/getFog", response_model=list[Fog])
async def get_domains(request: Request):
    service = request.app.state.fog_service
    return await service.get_fog()


@router.get("/getVitals", response_model=FogVitals)
async def get_domains(request: Request, fog_id: int):
    service = request.app.state.fog_service
    return await service.get_vitals(fog_id)

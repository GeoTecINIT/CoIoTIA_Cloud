from typing import Any

from fastapi import APIRouter, Depends, Request
from api.auth import get_token

from model.Coi import Coi

router = APIRouter()

@router.post("/createInUseCase")
async def create_devices_use_case(request: Request, use_case_id: int, geojson: dict[str, Any], token: str = Depends(get_token)):
    service = request.app.state.coi_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    geometries = [
        feature["geometry"]
        for feature in geojson["features"]
    ]
    await service.create_cois_use_case(geometries, use_case_id)


@router.get("/list", response_model=list[Coi])
async def get_cois_use_case(request: Request, use_case_id: int):
    service = request.app.state.coi_service
    return await service.list_cois_use_case(use_case_id)
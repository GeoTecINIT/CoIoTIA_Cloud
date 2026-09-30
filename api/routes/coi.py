from typing import Any

from fastapi import APIRouter, Depends, Request, Response, Header
from fastapi.responses import JSONResponse
from api.auth import get_token
import json

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
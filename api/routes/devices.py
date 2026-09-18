from fastapi import APIRouter, Depends, Request, Response, Header
from fastapi.responses import JSONResponse
from api.auth import get_token

from model.Device import DeviceCreate

router = APIRouter()

@router.get("/list")
async def get_devices(request: Request, token: str = Depends(get_token)):
    service = request.app.state.device_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    devices = await service.get_devices(uid)
    return devices

@router.get("/listUseCase")
async def get_devices_of_use_case(request: Request, use_case_id: int, token: str = Depends(get_token)):
    service = request.app.state.device_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    devices = await service.get_devices_of_use_case(use_case_id, uid)
    return devices

@router.post("/createInUseCase")
async def create_devices_use_case(request: Request, payload: list[DeviceCreate], token: str = Depends(get_token)):
    service = request.app.state.device_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    await service.create_devices_use_case(payload, uid)
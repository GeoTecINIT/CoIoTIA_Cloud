from fastapi import APIRouter, Depends, Request, Header
from api.auth import get_token

import httpx

from api.utils import forward_request

from model.Device import Device, DeviceRead

router = APIRouter()

@router.get("/list")
async def get_devices(request: Request, token: str = Depends(get_token)):
    service = request.app.state.device_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    devices = await service.get_devices(uid)
    return devices

@router.get("/listUseCase", response_model=list[DeviceRead])
async def get_devices_of_use_case(request: Request, use_case_id: int, token: str = Depends(get_token)):
    service = request.app.state.device_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    devices = await service.get_devices_of_use_case(use_case_id, uid)
    return devices

@router.get("/getDevice", response_model=DeviceRead)
async def get_device(request: Request, mac: str, token: str = Depends(get_token)):
    service = request.app.state.device_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    device = await service.get_device(mac, uid)
    return device

@router.post("/createInUseCase")
async def create_devices_use_case(request: Request, payload: list[Device], token: str = Depends(get_token)):
    device_service = request.app.state.device_service
    coi_service = request.app.state.coi_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    use_case_id = payload[0].use_case_id
    await device_service.create_devices_use_case(payload, uid)
    await coi_service.set_cois_of_devices_use_case(use_case_id)
    await device_service.create_virtual_devices(use_case_id, uid, request.app.state.production)


@router.post("/setStatus")
async def set_status(request: Request):
    data = await request.json()
    service = request.app.state.device_service
    await service.set_status(data["mac"], data["status"])


@router.post("/setVitals")
async def set_vitals(request: Request):
    data = await request.json()
    service = request.app.state.device_service
    await service.set_vitals(data["mac"], data["total_ram"], data["used_ram"])


@router.post("/setPosition")
async def set_position(request: Request):
    data = await request.json()
    service = request.app.state.device_service
    await service.set_vitals(data["mac"], data["lat"], data["lon"])


@router.get("/lastInference")
async def get_last_inference(request: Request, x_target_ip: str = Header(...), token: str = Depends(get_token)):
    return await forward_request("inferences/lastInference", request, x_target_ip)


@router.delete("/deleteInUseCase")
async def delete_devices_use_case(request: Request, use_case_id: int):
    service = request.app.state.device_service
    await service.delete_devices_use_case(use_case_id)


@router.post("/start")
async def start(request: Request, mac: str, x_target_ip: str = Header(...), token: str = Depends(get_token)):
    service = request.app.state.device_service
    name = await service.set_status(mac, "running")
    httpx.post(f"{x_target_ip}/virtual/start", params={"name" : name.replace(" ", "").lower()})


@router.post("/stop")
async def stop(request: Request, mac: str, x_target_ip: str = Header(...), token: str = Depends(get_token)):
    service = request.app.state.device_service
    name = await service.set_status(mac, "stopped")
    httpx.post(f"{x_target_ip}/virtual/stop", params={"name" : name.replace(" ", "").lower()})
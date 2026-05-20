from fastapi import APIRouter, Request, Header
from fastapi.responses import JSONResponse
import httpx
import asyncio

from api.utils import forward_request, build_fog_url

router = APIRouter()


@router.post("/list")
async def list_federated_devices(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("federated/list", request, x_target_ip)


@router.get("/listAll")
async def list_all_federated_devices(request: Request):
    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(*[
            client.post(
                build_fog_url(data["ip"], "federated/list"),
                data={}
            )
            for data in request.app.state.fog_devices.values()
            if data.get("status") == 1
        ], return_exceptions=True)

    devices = []
    for r in responses:
        if isinstance(r, Exception):
            continue
        try:
            devices.extend(r.json())
        except Exception:
            continue

    return JSONResponse(devices)
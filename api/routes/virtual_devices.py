from fastapi import APIRouter, Request, Response, Header
from fastapi.responses import JSONResponse
import httpx
import json

import utils
import firebase_utils

router = APIRouter()

async def forward_request(path: str, request: Request, x_target_ip: str):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/{path}",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)


@router.post("/createAll")
async def create_all_virtual_devices(request: Request):
    form = await request.form()
    user = form.get("user")
    domain = form.get("domain")
    sensors = json.loads(form.get("sensors"))
    virtual_type = form.get("virtual_type")

    sensors_per_cluster = utils.group_sensors_by_region(sensors)
    fog_region = firebase_utils.get_fog_of_regions(user, domain)

    for key, value in fog_region.items():
        fog_name = request.app.state.fog_id_name[value]
        fog_region[key] = request.app.state.fog_devices[fog_name]["ip"]

    async with httpx.AsyncClient() as client:
        for region, sensors_json in sensors_per_cluster.items():
            ip = fog_region.get(region)

            if not ip:
                continue

            await client.post(
                f"http://{ip}:5000/createVirtualDevices",
                data={
                    "user": user,
                    "sensors": json.dumps(sensors_json),
                    "virtual_type": virtual_type
                }
            )

    return JSONResponse(
        content={"message": "Virtual devices creation initiated"},
        status_code=200
    )

@router.post("/list")
async def list_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("listVirtualDevices", request, x_target_ip)

@router.post("/create")
async def create_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("createVirtualDevices", request, x_target_ip)

@router.post("/delete")
async def delete_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("deleteVirtualDevices", request, x_target_ip)

@router.post("/start")
async def start_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("startVirtualDevices", request, x_target_ip)

@router.post("/stop")
async def stop_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("stopVirtualDevices", request, x_target_ip)
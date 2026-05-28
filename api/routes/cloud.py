from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import json
import os

import utils

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@router.get("/getDevices")
async def get_devices(request: Request):
    queue = request.app.state.event_queue
    fog_devices = request.app.state.fog_devices

    async def event_stream():
        initial_data = {
            device: {
                "id" : data["id"],
                "ip" : data["ip"],
                "status" : data["status"]
            } for device, data in fog_devices.items()
        }
        yield f"data: {json.dumps(initial_data)}\n\n"
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type='text/event-stream')


@router.get("/getMetrics")
async def get_metrics(request: Request, device: str):
    if not device or device not in request.app.state.fog_devices:
        return JSONResponse(content={"error": "Invalid or missing device"}, status_code=400)
    fog_info = request.app.state.fog_devices[device]
    metrics = {
        "CPU": fog_info.get("cpu"),
        "RAM": fog_info.get("ram"),
        "Disk": fog_info.get("disk")
    }
    return JSONResponse(content=metrics, status_code=200)


@router.get("/listTemplates")
async def list_templates():
    with open(os.path.join(BASE_DIR, "templates", "templates.json")) as f:
        templates = json.load(f)
    return JSONResponse(content=templates, status_code=200)


@router.get("/listTypes")
async def list_types():
    with open(os.path.join(BASE_DIR, "virtual_device_types.json")) as f:
        types = json.load(f)
    return JSONResponse(content=types, status_code=200)


@router.post("/determineRegion")
async def determine_region(regions: str = Form(...), sensors: str = Form(...)):
    regions_data = json.loads(regions)
    sensors_data = json.loads(sensors)
    sensor_regions = utils.determine_sensor_region(regions_data, sensors_data)
    return JSONResponse(json.loads(sensor_regions))
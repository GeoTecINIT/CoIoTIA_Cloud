from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import json

import utils

router = APIRouter()

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


@router.post("/determineRegion")
async def determine_region(request: Request, regions: UploadFile = File(...), sensors: UploadFile = File(...)):
    form = await request.form()
    regions = json.loads(form.get("regions"))
    sensors = json.loads(form.get("sensors"))
    sensor_regions = utils.determine_sensor_region(regions, sensors)
    return sensor_regions
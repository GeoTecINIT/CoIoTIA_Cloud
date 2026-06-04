from fastapi import APIRouter, Request, Response, Header
from fastapi.responses import JSONResponse
import httpx
import asyncio
import json

import utils
import firebase_utils
from api.utils import build_fog_url

router = APIRouter()


@router.post("/createAll")
async def create_all_virtual_devices(request: Request):
    form = await request.form()
    user = form.get("user")
    domain = form.get("domain")
    sensors = json.loads(form.get("sensors"))
    node_type = form.get("node_type", "traditional")
    virtual_type = form.get("virtual_type", node_type)

    cluster_id = form.get("cluster_id")
    mqtt_broker_host = form.get("mqtt_broker_host")
    mqtt_broker_port = form.get("mqtt_broker_port")
    mqtt_network = form.get("mqtt_network")
    image = form.get("image")
    sensor_api_base_url = form.get("sensor_api_base_url")
    start = form.get("start", "false")

    sensors_per_cluster = utils.group_sensors_by_region(sensors)
    fog_region = firebase_utils.get_fog_of_regions(user, domain)

    for key, value in fog_region.items():
        fog_name = request.app.state.fog_id_name[value]
        fog_region[key] = request.app.state.fog_devices[fog_name]["ip"]

    tasks = []
    async with httpx.AsyncClient() as client:
        for region, sensors_json in sensors_per_cluster.items():
            ip = fog_region.get(region)
            if not ip:
                continue

            if node_type == "federated":
                for sensor in sensors_json:
                    sensor_form = {
                        "user_uid": user,
                        "name": sensor.get("id", sensor.get("name")),
                        "analysis_type": sensor.get("analysis_type", "temperature"),
                        "data_type": sensor.get("data_type", "sensor_data"),
                        "lat": str(sensor.get("lat", "")),
                        "lon": str(sensor.get("lon", "")),
                        "node_type": "federated",
                        "sensor_id": sensor.get("id", sensor.get("name")),
                    }
                    if cluster_id:
                        sensor_form["cluster_id"] = cluster_id
                    if mqtt_broker_host:
                        sensor_form["mqtt_broker_host"] = mqtt_broker_host
                    if mqtt_broker_port:
                        sensor_form["mqtt_broker_port"] = mqtt_broker_port
                    if mqtt_network:
                        sensor_form["mqtt_network"] = mqtt_network
                    if image:
                        sensor_form["image"] = image
                    if sensor_api_base_url:
                        sensor_form["sensor_api_base_url"] = sensor_api_base_url
                    sensor_form["start"] = start

                    tasks.append(client.post(
                        build_fog_url(ip, "virtual/create"),
                        data=sensor_form
                    ))
            else:
                tasks.append(client.post(
                    build_fog_url(ip, "virtual/create"),
                    data={
                        "user_uid": user,
                        "sensors": json.dumps(sensors_json),
                        "virtual_type": virtual_type,
                    }
                ))

        await asyncio.gather(*tasks, return_exceptions=True)

    return JSONResponse(
        content={"message": "Virtual devices creation initiated"},
        status_code=200
    )


@router.post("/list")
async def list_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/virtual/list",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)


@router.post("/create")
async def create_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    form = await request.form()
    request.app.state.logger.info("Creating virtual device")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/virtual/create",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)


@router.post("/delete")
async def delete_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/virtual/delete",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)


@router.post("/start")
async def start_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/virtual/start",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)


@router.post("/stop")
async def stop_virtual_devices(request: Request, x_target_ip: str = Header(...)):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/virtual/stop",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)


@router.post("/online")
async def get_online_devices(request: Request, x_target_ip: str = Header(...)):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/virtual/online",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)
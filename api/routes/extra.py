from fastapi import APIRouter, Request, Response, Header
import httpx
import asyncio

from api.utils import build_fog_url

router = APIRouter()


@router.get("/getDeploymentModels")
async def get_deployment_models(request: Request):
    deployment = request.query_params.get("deployment")

    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(*[
            client.get(
                build_fog_url(data["ip"], "getDeploymentModels"),
                params={"deployment": deployment}
            )
            for data in request.app.state.fog_devices.values()
        ])

    models = []
    for r in responses:
        models.extend(r.json())

    return models


@router.get("/getCode")
async def get_code(request: Request, x_target_ip: str = Header(...)):
    query_string = str(request.url.query)
    url = f"http://{x_target_ip}/getCode"
    if query_string:
        url = f"{url}?{query_string}"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)

    return Response(content=resp.content, status_code=resp.status_code)


@router.post("/getOnlineDevices")
async def get_online_devices(request: Request, x_target_ip: str = Header(...)):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/virtual/online",
            data=form,
        )

    return Response(content=resp.content, status_code=resp.status_code)
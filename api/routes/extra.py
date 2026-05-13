from fastapi import APIRouter, Request, Response
import httpx
import asyncio

router = APIRouter()

async def forward_request(path: str, request: Request, x_target_ip: str, files=None):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://{x_target_ip}/{path}",
            data=form,
            files=files
        )

    return Response(content=resp.content, status_code=resp.status_code)


@router.get("/getDeploymentModels")
async def get_deployment_models(request: Request):
    deployment = request.query_params.get("deployment")

    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(*[
            client.get(
                f"http://{data['ip']}:5000/getDeploymentModels",
                params={"deployment": deployment}
            )
            for data in request.app.state.fog_devices.values()
        ])

    models = []
    for r in responses:
        models.extend(r.json())

    return models

@router.get("/getCode")
async def get_code(request: Request, x_target_ip: str):
    return await forward_request("getCode", request, x_target_ip)

@router.post("/getOnlineDevices")
async def get_online_devices(request: Request, x_target_ip: str):
    return await forward_request("getOnlineDevices", request, x_target_ip)

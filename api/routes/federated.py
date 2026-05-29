from fastapi import APIRouter, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
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


@router.post("/server/start")
async def server_start(x_target_ip: str = Header(...)):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(f"http://{x_target_ip}/federated/server/start")
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception as exc:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)


@router.post("/server/stop")
async def server_stop(x_target_ip: str = Header(...)):
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"http://{x_target_ip}/federated/server/stop")
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception as exc:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)


@router.get("/server/status")
async def server_status(fog_ip: str):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"http://{fog_ip}/federated/server/status",
                timeout=5.0,
            )
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception as exc:
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=503)


async def _log_sse(request: Request, fog_ip: str, container_name: str):
    async def event_stream():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
                async with client.stream(
                    "GET",
                    f"http://{fog_ip}/federated/logs/stream",
                    params={"name": container_name},
                ) as fog_response:
                    async for chunk in fog_response.aiter_raw():
                        if await request.is_disconnected():
                            return
                        yield chunk
        except Exception as exc:
            import json
            payload = json.dumps({"ts": "", "line": f"[proxy error: {exc}]", "stream": "stderr"})
            yield f"data: {payload}\n\n".encode()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/logs/server")
async def server_logs_sse(request: Request, fog_ip: str):
    return await _log_sse(request, fog_ip, "mqtt-fl-server")


@router.get("/logs/node")
async def node_logs_sse(request: Request, fog_ip: str, sensor_id: str):
    return await _log_sse(request, fog_ip, f"mqtt-fl-node-{sensor_id}")

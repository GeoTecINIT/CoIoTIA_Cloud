from fastapi import Request, Response
import httpx


async def forward_request(path: str, request: Request, x_target_ip: str, files=None):
    form = await request.form()

    async with httpx.AsyncClient() as client:
        kwargs = {"data": form}
        if files:
            kwargs["files"] = files
        resp = await client.post(
            f"http://{x_target_ip}/{path}",
            **kwargs
        )

    return Response(content=resp.content, status_code=resp.status_code)


def build_fog_url(ip: str, path: str) -> str:
    return f"http://{ip}:5000/{path}"
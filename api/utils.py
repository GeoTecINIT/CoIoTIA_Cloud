from fastapi import Request, Response
import httpx


async def forward_request(path: str, request: Request, x_target_ip: str, files=None, uid: str = None):
    internal_headers = {}
    if uid:
        internal_headers["X-User-UID"] = uid

    async with httpx.AsyncClient() as client:
        if request.method == "GET":
            resp = await client.get(
                f"http://{x_target_ip}/{path}",
                params=dict(request.query_params),
                headers=internal_headers
            )
        else:
            form = await request.form()

            form_data = {
                key: value
                for key, value in form.multi_items()
                if not hasattr(value, "read")
            }

            kwargs = {"data": form_data}
            if files:
                kwargs["files"] = files
            resp = await client.post(
                f"http://{x_target_ip}/{path}",
                **kwargs,
                headers=internal_headers
            )

    return Response(content=resp.content, status_code=resp.status_code)


def build_fog_url(ip: str, path: str) -> str:
    return f"http://{ip}:5000/{path}"
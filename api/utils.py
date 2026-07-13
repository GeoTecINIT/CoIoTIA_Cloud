from fastapi import Request, Response, HTTPException
import httpx


async def forward_request(path: str, request: Request, x_target_ip: str, files=None, uid: str = None):
    internal_headers = {}
    if uid:
        internal_headers["X-User-UID"] = uid

    async with httpx.AsyncClient() as client:
        if request.method == "GET":
            try:
                resp = await client.get(
                    f"http://{x_target_ip}/{path}",
                    params=dict(request.query_params),
                    headers=internal_headers,
                    timeout=None
                )
            except httpx.ConnectTimeout:
                raise HTTPException(status_code=504, detail="Timeout connecting to fog")
            except httpx.ConnectError:
                raise HTTPException(status_code=503, detail="Can't connect to fog")
            except:
                raise HTTPException(status_code=500, detail="Unexpected error")

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

            try: 
                resp = await client.post(
                    f"http://{x_target_ip}/{path}",
                    **kwargs,
                    headers=internal_headers
                )
            except httpx.ConnectTimeout:
                raise HTTPException(status_code=504, detail="Timeout connecting to fog")
            except httpx.ConnectError:
                raise HTTPException(status_code=503, detail="Can't connect to fog")
            except:
                raise HTTPException(status_code=500, detail="Unexpected error")

    return Response(content=resp.content, status_code=resp.status_code)


def build_fog_url(ip: str, path: str) -> str:
    return f"http://{ip}:5000/{path}"
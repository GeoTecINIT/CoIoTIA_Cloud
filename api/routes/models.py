from fastapi import APIRouter, Header, File, Request, Response, UploadFile
import httpx

from api.utils import forward_request

router = APIRouter()


@router.post("/list")
async def list_models(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("list", request, x_target_ip)

@router.post("/upload")
async def upload_model(request: Request, x_target_ip: str = Header(...), files: list[UploadFile] = File(...)):
    files_to_forward = []
    for f in files:
        content = await f.read()
        files_to_forward.append((f.filename, content, f.content_type))
    return await forward_request("upload", request, x_target_ip, files=files_to_forward)

@router.post("/update")
async def update_model(request: Request, x_target_ip: str = Header(...), files: list[UploadFile] = File(...)):
    files_to_forward = []
    for f in files:
        content = await f.read()
        files_to_forward.append((f.filename, content, f.content_type))
    return await forward_request("update", request, x_target_ip, files=files_to_forward)

@router.post("/delete")
async def delete_model(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("delete", request, x_target_ip)

@router.post("/deploy")
async def deploy_model(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("deploy", request, x_target_ip)
from fastapi import APIRouter, Header, File, Request, Response, UploadFile
import httpx

from api.utils import forward_request
import firebase_utils

router = APIRouter()


@router.get("/list")
async def list_models(request: Request, x_target_ip: str = Header(...), authorization: str = Header(...)):
    uid = firebase_utils.verify_firebase_token(authorization)
    return await forward_request("list", request, x_target_ip, uid=uid)

@router.post("/upload")
async def upload_model(request: Request, x_target_ip: str = Header(...), model: UploadFile = File(...), code: UploadFile = File(...)):
    model_content = await model.read()
    code_content = await code.read()
    files_to_forward = [
        ("model", (model.filename, model_content, model.content_type)),
        ("code", (code.filename, code_content, code.content_type))
    ]
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
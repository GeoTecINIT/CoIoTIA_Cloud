from fastapi import APIRouter, Depends, Header, File, Request, Response, UploadFile
from api.auth import get_token

from api.utils import forward_request

router = APIRouter()


@router.get("/list")
async def list_models(request: Request, x_target_ip: str = Header(...), token: str = Depends(get_token)):
    uid = request.app.state.firebase.verify_firebase_token(token)
    return await forward_request("models/list", request, x_target_ip, uid=uid)

@router.post("/upload")
async def upload_model(request: Request, x_target_ip: str = Header(...), token: str = Depends(get_token), model: UploadFile = File(...), code: UploadFile = File(...)):
    uid = request.app.state.firebase.verify_firebase_token(token)
    model_content = await model.read()
    code_content = await code.read()
    files_to_forward = [
        ("model", (model.filename, model_content, model.content_type)),
        ("code", (code.filename, code_content, code.content_type))
    ]
    return await forward_request("models/upload", request, x_target_ip, files=files_to_forward, uid=uid)

@router.put("/update")
async def update_model(request: Request, x_target_ip: str = Header(...), model: UploadFile = File(...), code: UploadFile = File(...)):
    files_to_forward = [
        ("model", (
            model.filename,
            await model.read(),
            model.content_type,
        )),
        ("code", (
            code.filename,
            await code.read(),
            code.content_type,
        )),
    ]
    return await forward_request("models/update", request, x_target_ip, files=files_to_forward)

@router.delete("/delete")
async def delete_model(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("models/delete", request, x_target_ip)

@router.post("/deploy")
async def deploy_model(request: Request, x_target_ip: str = Header(...)):
    return await forward_request("models/deploy", request, x_target_ip)
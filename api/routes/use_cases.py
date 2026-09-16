from fastapi import APIRouter, Depends, Request, Response, Header
from fastapi.responses import JSONResponse
from api.auth import get_token

from model.UseCase import UseCaseCreate, UseCaseCreateResponse

router = APIRouter()

@router.get("/list")
async def get_use_cases(request: Request, token: str = Depends(get_token)):
    service = request.app.state.use_case_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    use_cases = await service.get_use_cases(uid)
    return use_cases

@router.post("/create", response_model=UseCaseCreateResponse)
async def create_use_case(request: Request, payload: UseCaseCreate, token: str = Depends(get_token)):
    service = request.app.state.use_case_service
    uid = request.app.state.firebase.verify_firebase_token(token)
    id = await service.create_use_case(payload, uid)
    return {"id": id}
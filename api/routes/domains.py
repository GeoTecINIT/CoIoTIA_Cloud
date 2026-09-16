from fastapi import APIRouter, Depends, Request, Response, Header
from fastapi.responses import JSONResponse

from model.Domain import Domain

router = APIRouter()

@router.get("/getDomains", response_model=list[Domain])
async def get_domains(request: Request):
    service = request.app.state.domain_service
    return await service.get_domains()
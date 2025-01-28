from fastapi import APIRouter
from . import registration, login, token, email

auth_router = APIRouter(prefix="/auth", tags=["authentication"])

auth_router.include_router(registration.router)
auth_router.include_router(login.router)
auth_router.include_router(token.router)

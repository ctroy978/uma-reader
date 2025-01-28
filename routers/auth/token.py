from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Annotated

from database.session import get_db
from verification.session import get_verification_db
from database.models import User
from database.models.token import RefreshToken
from verification.models import TokenBlacklist
from auth.middleware import require_auth

from auth.jwt_utils import (
    create_access_token,
    create_refresh_token,
    verify_token,
    blacklist_token,
    get_token_from_header,
)
from auth.jwt_config import get_jwt_settings, get_token_settings

settings = get_jwt_settings()
token_settings = get_token_settings()

router = APIRouter(tags=["Authentication"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str


@router.post("/token/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    verification_db: Annotated[Session, Depends(get_verification_db)],
):
    """Generate new access token using refresh token"""
    refresh_token = request.cookies.get(settings.COOKIE_NAME)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token provided")

    # Verify refresh token
    payload = verify_token(refresh_token, verification_db, "refresh")
    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Generate new tokens
    access_token = create_access_token(user)
    new_refresh_token = create_refresh_token(user, db, request)

    # Create response with new tokens
    token_response = TokenResponse(
        access_token=access_token, token_type=settings.TOKEN_TYPE, role=user.role_name
    )

    response = Response(content=token_response.model_dump_json())
    response.set_cookie(**token_settings["COOKIE_SETTINGS"], value=new_refresh_token)

    return response


@router.post("/token/revoke")
async def revoke_token(
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    verification_db: Annotated[Session, Depends(get_verification_db)],
):
    """Revoke access and refresh tokens"""
    access_token = get_token_from_header(request)
    if access_token:
        blacklist_token(access_token, verification_db, "access")

    refresh_token = request.cookies.get(settings.COOKIE_NAME)
    if refresh_token:
        blacklist_token(refresh_token, verification_db, "refresh")

    response.delete_cookie(
        settings.COOKIE_NAME,
        domain=token_settings["COOKIE_SETTINGS"]["domain"],
        path=token_settings["COOKIE_SETTINGS"]["path"],
    )

    return {"message": "Tokens revoked successfully"}


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(require_auth),
    db: Session = Depends(get_db),
    verification_db: Session = Depends(get_verification_db),
):
    """Log out user and revoke all their tokens"""
    try:
        # Revoke all refresh tokens for user
        active_tokens = (
            db.query(RefreshToken)
            .filter(
                RefreshToken.user_id == current_user.id,
                RefreshToken.revoked_at.is_(None),
            )
            .all()
        )

        for token in active_tokens:
            token.revoke()

        db.commit()

        # Blacklist current access token
        access_token = get_token_from_header(request)
        if access_token:
            blacklist_token(access_token, verification_db, "access")

        # Clear refresh token cookie
        response.delete_cookie(
            settings.COOKIE_NAME,
            domain=token_settings["COOKIE_SETTINGS"]["domain"],
            path=token_settings["COOKIE_SETTINGS"]["path"],
        )

        return {"message": "Successfully logged out"}

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during logout: {str(e)}",
        )

# routers/auth/logout.py
from fastapi import APIRouter, Depends, Response, Request
from sqlalchemy.orm import Session
from database.session import get_db
from verification.session import get_verification_db
from auth.jwt_utils import get_token_from_header, blacklist_token

router = APIRouter(tags=["Authentication"])


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    verification_db: Session = Depends(get_verification_db),
):
    # Get the access token from header
    token = get_token_from_header(request)

    if token:
        # Blacklist the current access token
        blacklist_token(token, verification_db)

    # Clear the refresh token cookie
    response.delete_cookie(
        key="refresh_token", httponly=True, secure=True, samesite="lax"
    )

    return {"message": "Successfully logged out"}

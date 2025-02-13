# logout.py
from fastapi import APIRouter, Depends, Response, Request, Cookie, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone

from database.session import get_db
from verification.session import get_verification_db
from database.models import RefreshToken
from auth.jwt_utils import get_token_from_header, blacklist_token

router = APIRouter(tags=["Authentication"])


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    refresh_token: Optional[str] = Cookie(None, alias="refresh_token"),
    db: Session = Depends(get_db),
    verification_db: Session = Depends(get_verification_db),
):
    """
    Logout endpoint that works regardless of JWT status.
    Uses refresh token cookie to identify and terminate session.
    """
    # Get the access token if present (but don't require it)
    try:
        token = get_token_from_header(request)
        if token:
            # Best effort to blacklist the access token if it's valid
            blacklist_token(token, verification_db)
    except:
        # Ignore any token validation errors
        pass

    # Revoke refresh token if present
    if refresh_token:
        try:
            # Find and revoke the refresh token
            token_record = (
                db.query(RefreshToken)
                .filter(
                    RefreshToken.token == refresh_token,
                    RefreshToken.revoked_at.is_(None),
                )
                .first()
            )

            if token_record:
                token_record.revoked_at = datetime.now(timezone.utc)
                db.commit()
        except:
            # Continue even if database update fails
            pass

    # Always clear the refresh token cookie
    response.delete_cookie(
        key="refresh_token", httponly=True, secure=True, samesite="lax"
    )

    return {"message": "Successfully logged out"}


# auth/middleware.py
from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Optional
from functools import wraps

from database.session import get_db
from database.models import User
from auth.jwt_utils import decode_token, get_token_from_header


async def get_current_user(
    request: Request, required: bool = True, db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Modified middleware that can optionally require authentication.
    """
    try:
        token = get_token_from_header(request)
        if not token and required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
            )
        elif not token:
            return None

        payload = decode_token(token)
        user = db.query(User).filter(User.id == payload["sub"]).first()

        if not user and required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )
        return user
    except HTTPException:
        if required:
            raise
        return None


# Modify the existing require_user dependency to use get_current_user
require_user = lambda: get_current_user(required=True)
optional_user = lambda: get_current_user(required=False)


# auth/jwt_utils.py (Add new function)
def blacklist_token_safe(token: str, verification_db: Session) -> bool:
    """
    Safely blacklist a token, returning False if token is invalid.
    """
    try:
        blacklist_token(token, verification_db)
        return True
    except:
        return False

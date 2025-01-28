from typing import Optional, List
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from database.session import get_db
from verification.session import get_verification_db
from database.models import User
from auth.jwt_utils import verify_token, get_token_from_header


class AuthMiddleware:
    security = HTTPBearer()

    def __init__(self, allowed_roles: Optional[List[str]] = None):
        self.allowed_roles = allowed_roles

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db),
        verification_db: Session = Depends(get_verification_db),
    ) -> User:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No credentials provided",
            )

        try:
            payload = verify_token(credentials.credentials, verification_db, "access")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            )

        user = db.query(User).filter(User.id == payload["sub"]).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

        if user.is_deleted:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated",
            )

        if self.allowed_roles and user.role_name not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {user.role_name} not permitted",
            )

        request.state.user = user
        return user


async def get_optional_user(
    request: Request,
    db: Session = Depends(get_db),
    verification_db: Session = Depends(get_verification_db),
) -> Optional[User]:
    token = get_token_from_header(request)
    if not token:
        return None

    try:
        payload = verify_token(token, verification_db, "access")
    except:
        return None

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user or user.is_deleted:
        return None

    request.state.user = user
    return user


# Role-based middleware instances
require_user = AuthMiddleware(["STUDENT", "TEACHER", "ADMIN"])
require_teacher = AuthMiddleware(["TEACHER", "ADMIN"])
require_admin = AuthMiddleware(["ADMIN"])
require_auth = AuthMiddleware()  # Any authenticated user
get_current_user = get_optional_user  # Alias for optional auth

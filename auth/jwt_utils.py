from datetime import datetime, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from uuid import uuid4
from fastapi import HTTPException, status, Request
from sqlalchemy.orm import Session

from .jwt_config import get_jwt_settings, get_token_settings
from database.models import User, RefreshToken
from verification.models import TokenBlacklist


settings = get_jwt_settings()
token_settings = get_token_settings()


def create_token_payload(user: User, token_type: str = "access") -> Dict[str, Any]:
    """Create the JWT payload for a user"""
    jti = str(uuid4())
    now = datetime.now(timezone.utc)

    expires_delta = (
        token_settings["ACCESS_EXPIRES"]
        if token_type == "access"
        else token_settings["REFRESH_EXPIRES"]
    )

    expires_at = now + expires_delta

    payload = {
        "sub": str(user.id),
        "role": user.role_name,
        "type": token_type,
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }

    return payload


def create_access_token(user: User) -> str:
    """Generate a new access token"""
    payload = create_token_payload(user, "access")
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(
    user: User, db: Session, request: Request, invalidate_existing: bool = True
) -> str:
    """Generate a new refresh token and store it"""
    # Optionally invalidate existing refresh tokens
    if invalidate_existing:
        existing_tokens = (
            db.query(RefreshToken)
            .filter(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
            .all()
        )
        for token in existing_tokens:
            token.revoke()

    # Create new token
    payload = create_token_payload(user, "refresh")
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    # Store token
    db_token = RefreshToken(
        token=token,
        user_id=user.id,
        expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        issued_by_ip=request.client.host,
        browser_info=request.headers.get("user-agent"),
    )

    db.add(db_token)
    db.commit()

    return token


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_token(
    token: str, verification_db: Session, expected_type: str = "access"
) -> Dict[str, Any]:
    """Verify a token's validity and type"""
    payload = decode_token(token)

    # Check token type
    if payload.get("type") != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token type. Expected {expected_type}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token is blacklisted
    if TokenBlacklist.is_blacklisted(verification_db, payload["jti"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def blacklist_token(
    token: str, verification_db: Session, token_type: str = "access"
) -> None:
    """Add a token to the blacklist"""
    try:
        payload = decode_token(token)
        if payload.get("type") != token_type:
            return  # Only blacklist tokens of the specified type

        blacklist_entry = TokenBlacklist(
            jti=payload["jti"],
            expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )

        verification_db.add(blacklist_entry)
        verification_db.commit()

    except HTTPException:
        pass  # Invalid tokens don't need to be blacklisted


def get_token_from_header(request: Request) -> Optional[str]:
    """Extract token from Authorization header"""
    auth = request.headers.get("Authorization")
    if not auth:
        return None

    parts = auth.split()
    if parts[0].lower() != "bearer" or len(parts) != 2:
        return None

    return parts[1]

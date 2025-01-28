from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Response,
    Request,
    Response,
)
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Annotated
import random
import string

from database.session import get_db
from verification.session import get_verification_db
from database.models import User
from verification.models import OTPVerification, VerificationType
from .email import send_login_email
from verification.models import OTPVerification, VerificationType
from auth.jwt_utils import create_access_token, create_refresh_token

router = APIRouter(tags=["Authentication"])


class LoginRequest(BaseModel):
    email: EmailStr

    class Config:
        json_schema_extra = {"example": {"email": "user@example.com"}}


class LoginVerification(BaseModel):
    email: EmailStr
    verification_code: str

    class Config:
        json_schema_extra = {
            "example": {"email": "user@example.com", "verification_code": "123456"}
        }


@router.post("/login/initiate")
async def initiate_login(
    login: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
    verification_db: Annotated[Session, Depends(get_verification_db)],
):
    # Check user exists
    user = db.query(User).filter(User.email == login.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No account found with this email",
        )

    # Generate and store OTP
    otp = "".join(random.choices(string.digits, k=6))
    verification = OTPVerification(
        email=login.email,
        verification_code=otp,
        verification_type=VerificationType.LOGIN,
    )
    verification_db.add(verification)
    verification_db.commit()

    # Send email
    await send_login_email(login.email, otp)

    return {"message": "Login verification code sent", "email": login.email}


@router.post("/login/verify")
async def verify_login(
    verification: LoginVerification,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    verification_db: Annotated[Session, Depends(get_verification_db)],
):
    # Verify OTP
    otp = (
        verification_db.query(OTPVerification)
        .filter(
            OTPVerification.email == verification.email,
            OTPVerification.verification_code == verification.verification_code,
            OTPVerification.verification_type == VerificationType.LOGIN,
            OTPVerification.is_used == False,  # noqa: E712
        )
        .first()
    )

    if not otp or not otp.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired verification code",
        )

    # Get user
    user = db.query(User).filter(User.email == verification.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Mark OTP as used
    otp.use_code()
    verification_db.commit()

    # Generate tokens
    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user, db, request)

    # Set refresh token in cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=30 * 24 * 60 * 60,  # 30 days
    )

    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "user_id": str(user.id),
        "role": user.role_name,
    }

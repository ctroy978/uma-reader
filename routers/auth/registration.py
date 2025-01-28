from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import os
from typing import Annotated
import random
import string

from database.session import get_db
from verification.session import get_verification_db
from database.models import User
from verification.models import OTPVerification, VerificationType
from .email import send_registration_email
from auth.jwt_utils import create_access_token

router = APIRouter(tags=["Authentication"])


class InitialRegistration(BaseModel):
    username: str
    email: EmailStr
    full_name: str

    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "full_name": "John Doe",
            }
        }


class CompleteRegistration(BaseModel):
    username: str
    email: EmailStr
    verification_code: str
    full_name: str


@router.post("/register/initiate")
async def initiate_registration(
    registration: InitialRegistration,
    db: Annotated[Session, Depends(get_db)],
    verification_db: Annotated[Session, Depends(get_verification_db)],
):
    # Check existing email
    if db.query(User).filter(User.email == registration.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Check existing username
    if db.query(User).filter(User.username == registration.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
        )

    # Generate and store OTP
    otp = "".join(random.choices(string.digits, k=6))
    verification = OTPVerification(
        email=registration.email,
        verification_code=otp,
        verification_type=VerificationType.REGISTRATION,
    )
    verification_db.add(verification)
    verification_db.commit()

    # Send email
    await send_registration_email(registration.email, otp, registration.full_name)

    return {
        "message": "Registration verification code sent",
        "email": registration.email,
    }


@router.post("/register/complete")
async def complete_registration(
    registration: CompleteRegistration,
    db: Annotated[Session, Depends(get_db)],
    verification_db: Annotated[Session, Depends(get_verification_db)],
):
    # Verify OTP - original logic
    verification = (
        verification_db.query(OTPVerification)
        .filter(
            OTPVerification.email == registration.email,
            OTPVerification.verification_code == registration.verification_code,
            OTPVerification.verification_type == VerificationType.REGISTRATION,
            OTPVerification.is_used == False,  # noqa: E712
        )
        .first()
    )

    if not verification or not verification.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired verification code",
        )

    # Mark OTP as used
    verification.use_code()
    verification_db.commit()

    # Check if first admin
    is_admin = registration.email == os.getenv("INITIAL_ADMIN_EMAIL")
    role_name = "ADMIN" if is_admin else "STUDENT"

    # Create user
    new_user = User(
        username=registration.username,
        email=registration.email,
        full_name=registration.full_name,
        role_name=role_name,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Generate access token - only new code addition
    access_token = create_access_token(new_user)

    # Return response with token - modified return
    return {
        "message": "Registration successful",
        "user_id": new_user.id,
        "role": role_name,
        "access_token": access_token,
        "token_type": "Bearer",
    }

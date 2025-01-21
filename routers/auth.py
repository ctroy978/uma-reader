from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from typing import Annotated

from database.session import get_db
from verification.session import get_verification_db
from database.models import User, Role
from verification.models import OTPVerification, VerificationType

router = APIRouter(prefix="/auth", tags=["authentication"])


# Pydantic models for request validation
class InitialRegistration(BaseModel):
    username: str
    email: EmailStr
    full_name: str


class CompleteRegistration(BaseModel):
    username: str
    email: EmailStr
    verification_code: str


# Email configuration (move to config file in production)
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_USERNAME"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)

fastmail = FastMail(conf)


async def send_registration_email(email: str, otp: str, full_name: str):
    """Send registration verification email"""
    message = MessageSchema(
        subject="Complete Your Registration",
        recipients=[email],
        body=f"""
        <html>
            <body>
                <h2>Welcome to Student Reader, {full_name}!</h2>
                <p>Your registration verification code is: <strong>{otp}</strong></p>
                <p>This code will expire in 10 minutes.</p>
                <p>If you didn't request this registration, please ignore this email.</p>
            </body>
        </html>
        """,
        subtype="html",
    )
    await fastmail.send_message(message)


@router.post("/register/initiate")
async def initiate_registration(
    registration: InitialRegistration,
    db: Annotated[Session, Depends(get_db)],
    verification_db: Annotated[Session, Depends(get_verification_db)],
):
    """Start the registration process"""
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == registration.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Check if username already exists
    existing_username = (
        db.query(User).filter(User.username == registration.username).first()
    )
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
        )

    # Generate OTP
    import random
    import string

    otp = "".join(random.choices(string.digits, k=6))

    # Store OTP
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
    """Complete the registration process"""
    # Verify OTP
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

    # Check if this is the first admin
    is_admin = registration.email == os.getenv("FIRST_ADMIN_EMAIL")
    role_name = "ADMIN" if is_admin else "STUDENT"

    # Create user
    new_user = User(
        username=registration.username, email=registration.email, role_name=role_name
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "Registration successful",
        "user_id": new_user.id,
        "role": role_name,
    }

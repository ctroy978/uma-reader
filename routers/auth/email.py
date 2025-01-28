from fastapi import APIRouter
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import os
from dotenv import load_dotenv

load_dotenv()

email_config = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_USERNAME"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)

router = APIRouter(tags=["Authentication"])

mail_client = FastMail(email_config)


async def send_registration_email(email: str, otp: str, full_name: str):
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
    await mail_client.send_message(message)


async def send_login_email(email: str, otp: str):
    message = MessageSchema(
        subject="Login Verification Code",
        recipients=[email],
        body=f"""
        <html>
            <body>
                <h2>Login Verification Code</h2>
                <p>Your login verification code is: <strong>{otp}</strong></p>
                <p>This code will expire in 10 minutes.</p>
                <p>If you didn't request this login, please ignore this email.</p>
            </body>
        </html>
        """,
        subtype="html",
    )
    await mail_client.send_message(message)

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class JWTSettings(BaseSettings):
    # Required settings
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")

    # Optional settings with defaults
    ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "120")
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(
        os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30")
    )
    TOKEN_TYPE: str = os.getenv("JWT_TOKEN_TYPE", "Bearer")
    TOKEN_URL: str = os.getenv("JWT_TOKEN_URL", "/auth/token")
    COOKIE_NAME: str = os.getenv("JWT_COOKIE_NAME", "refresh_token")
    COOKIE_DOMAIN: Optional[str] = os.getenv("JWT_COOKIE_DOMAIN")
    COOKIE_PATH: str = os.getenv("JWT_COOKIE_PATH", "/")
    COOKIE_SECURE: bool = os.getenv("JWT_COOKIE_SECURE", "True").lower() == "true"
    COOKIE_HTTPONLY: bool = os.getenv("JWT_COOKIE_HTTPONLY", "True").lower() == "true"
    COOKIE_SAMESITE: str = os.getenv("JWT_COOKIE_SAMESITE", "lax")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


@lru_cache()
def get_jwt_settings() -> JWTSettings:
    """Get JWT settings with environment variables"""
    return JWTSettings()


def get_token_settings() -> Dict[str, Any]:
    """Get token-specific settings"""
    settings = get_jwt_settings()
    return {
        "ACCESS_EXPIRES": timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "REFRESH_EXPIRES": timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "COOKIE_SETTINGS": {
            "key": settings.COOKIE_NAME,
            "domain": settings.COOKIE_DOMAIN,
            "path": settings.COOKIE_PATH,
            "secure": settings.COOKIE_SECURE,
            "httponly": settings.COOKIE_HTTPONLY,
            "samesite": settings.COOKIE_SAMESITE,
        },
    }

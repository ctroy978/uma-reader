from datetime import datetime, timedelta, timezone
from sqlalchemy import Column, String, DateTime, Boolean, Enum
from sqlalchemy.orm import Mapped
import enum
from ..base import Base, TimestampMixin


class VerificationType(enum.Enum):
    """Type of verification being performed"""

    REGISTRATION = "registration"
    LOGIN = "login"


class OTPVerification(Base, TimestampMixin):
    """Stores temporary verification codes for both registration and login"""

    __tablename__ = "otp_verifications"

    # Core fields
    id: Mapped[str] = Column(String(36), primary_key=True)
    email: Mapped[str] = Column(String(255), nullable=False, index=True)
    verification_code: Mapped[str] = Column(String(8), nullable=False)
    verification_type: Mapped[VerificationType] = Column(
        Enum(VerificationType), nullable=False, index=True
    )
    expires_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    is_used: Mapped[bool] = Column(Boolean, default=False, nullable=False)

    def is_valid(self) -> bool:
        """Check if the verification code is still valid"""
        now = datetime.now(timezone.utc)
        return not self.is_used and now <= self.expires_at

    def use_code(self) -> None:
        """Mark the verification code as used"""
        self.is_used = True

    def __str__(self) -> str:
        status = "Valid" if self.is_valid() else "Invalid"
        return (
            f"{self.verification_type.value} verification for {self.email} ({status})"
        )

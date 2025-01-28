from datetime import datetime, timezone
from sqlalchemy import Column, String, ForeignKey, DateTime, Index, Boolean, Text
from sqlalchemy.orm import Mapped, relationship
from uuid import uuid4

from ..base import Base, TimestampMixin
from .user import User

# Update User model to include refresh tokens
User.refresh_tokens = relationship(
    "RefreshToken", back_populates="user", cascade="all, delete-orphan"
)


class RefreshToken(Base, TimestampMixin):
    """Stores refresh tokens for users"""

    __tablename__ = "refresh_tokens"

    # Core fields
    id: Mapped[str] = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    token: Mapped[str] = Column(String(255), unique=True, nullable=False, index=True)
    user_id: Mapped[str] = Column(
        String(36), ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )

    # Expiration and status
    expires_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)
    issued_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    revoked_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=True)

    # Audit info
    issued_by_ip: Mapped[str] = Column(String(45), nullable=False)  # IPv6 length
    browser_info: Mapped[str] = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="refresh_tokens")

    # Indexes
    __table_args__ = (Index("ix_refresh_tokens_user_expires", "user_id", "expires_at"),)

    def is_valid(self) -> bool:
        """Check if token is valid"""
        now = datetime.now(timezone.utc)
        return not self.revoked_at and now < self.expires_at

    def revoke(self) -> None:
        """Mark token as revoked"""
        self.revoked_at = datetime.now(timezone.utc)

    def __str__(self) -> str:
        status = "Valid" if self.is_valid() else "Invalid"
        return f"RefreshToken for user {self.user_id} ({status})"

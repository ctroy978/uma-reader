from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, Index
from sqlalchemy.orm import Mapped
from uuid import uuid4

from ..base import Base, TimestampMixin


class TokenBlacklist(Base, TimestampMixin):
    """Tracks revoked/blacklisted JWT tokens"""

    __tablename__ = "token_blacklist"

    # Core fields
    id: Mapped[str] = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    jti: Mapped[str] = Column(String(36), unique=True, nullable=False, index=True)
    blacklisted_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    expires_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)

    # Indexes
    __table_args__ = (Index("ix_token_blacklist_expires", "expires_at"),)

    @classmethod
    def is_blacklisted(cls, session, jti: str) -> bool:
        """Check if a token is blacklisted"""
        return (
            session.query(cls)
            .filter(cls.jti == jti, cls.expires_at > datetime.now(timezone.utc))
            .first()
            is not None
        )

    @classmethod
    def cleanup_expired(cls, session) -> int:
        """Remove expired blacklist entries"""
        result = (
            session.query(cls)
            .filter(cls.expires_at <= datetime.now(timezone.utc))
            .delete()
        )
        session.commit()
        return result

    def __str__(self) -> str:
        return f"Blacklisted token {self.jti} (expires: {self.expires_at})"

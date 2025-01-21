from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, DateTime


class Base(DeclarativeBase):
    """Base class for all verification models"""

    pass


class TimestampMixin:
    """Adds created_at and updated_at timestamps"""

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

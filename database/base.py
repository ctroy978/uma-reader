from datetime import datetime, timezone
import uuid
from typing import Any, Dict

from sqlalchemy import MetaData, DateTime, Boolean, Column, String
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import Mapped

# Configure metadata with naming conventions for SQLite
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


@as_declarative(metadata=metadata)
class Base:
    """Base class for all database models"""

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate tablename from class name"""
        return cls.__name__.lower()

    # Common columns that should appear in all tables
    id: Mapped[str] = Column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )


from datetime import datetime, timezone


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps"""

    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SoftDeleteMixin:
    """Mixin to add soft delete functionality"""

    is_deleted: Mapped[bool] = Column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    def soft_delete(self) -> None:
        """Mark record as deleted"""
        self.is_deleted = True

    @classmethod
    def not_deleted(cls) -> Any:
        """Query filter for non-deleted records"""
        return cls.is_deleted == False

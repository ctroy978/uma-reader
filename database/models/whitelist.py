# app/database/models/whitelist.py
from sqlalchemy import Column, String, Enum
import enum
from sqlalchemy.orm import Mapped, validates
import re

from ..base import Base, TimestampMixin, SoftDeleteMixin


class WhitelistType(str, enum.Enum):
    """Types of whitelist entries"""

    DOMAIN = "domain"
    EMAIL = "email"


class EmailWhitelist(Base, TimestampMixin, SoftDeleteMixin):
    """Whitelist for allowed email domains and specific emails"""

    # Core fields
    value: Mapped[str] = Column(String(255), unique=True, nullable=False, index=True)
    type: Mapped[WhitelistType] = Column(
        Enum(WhitelistType), nullable=False, default=WhitelistType.DOMAIN
    )
    description: Mapped[str] = Column(String(255), nullable=True)

    @validates("value")
    def validate_value(self, key, value):
        """Validate whitelist entry based on type"""
        if not value:
            raise ValueError("Whitelist value cannot be empty")

        # If already a model instance being loaded from DB
        if hasattr(self, "type"):
            whitelist_type = self.type
        else:
            # Default for new instances
            whitelist_type = WhitelistType.DOMAIN

        if whitelist_type == WhitelistType.DOMAIN:
            # Domain should be like: example.com, sub.example.com
            domain_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
            if not re.match(domain_pattern, value):
                raise ValueError("Invalid domain format")

        elif whitelist_type == WhitelistType.EMAIL:
            # Basic email format validation
            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, value):
                raise ValueError("Invalid email format")

        return value.lower()  # Store in lowercase for consistent matching

    def __str__(self) -> str:
        if self.type == WhitelistType.DOMAIN:
            return f"Allowed domain: {self.value}"
        else:
            return f"Allowed email: {self.value}"

# app/database/models/teacher_bypass.py
from sqlalchemy import Column, String, ForeignKey, Boolean, event
from sqlalchemy.orm import Mapped, relationship, validates
import re

from ..base import Base, TimestampMixin, SoftDeleteMixin


class TeacherBypassCode(Base, TimestampMixin, SoftDeleteMixin):
    """Stores bypass codes for teachers to help stuck students"""

    # Core fields
    teacher_id: Mapped[str] = Column(
        String(36), ForeignKey("user.id"), nullable=False, unique=True, index=True
    )
    bypass_code: Mapped[str] = Column(String(4), nullable=False)
    is_active: Mapped[bool] = Column(Boolean, nullable=False, default=True)

    # Relationships
    teacher = relationship("User", back_populates="bypass_code")

    @validates("bypass_code")
    def validate_bypass_code(self, key, bypass_code):
        """Validate bypass code format (4 digits)"""
        if not bypass_code:
            raise ValueError("Bypass code cannot be empty")

        # Bypass code format validation: exactly 4 digits
        pattern = r"^\d{4}$"
        if not re.match(pattern, bypass_code):
            raise ValueError("Bypass code must be exactly 4 digits")

        return bypass_code

    def __str__(self) -> str:
        return f"Bypass Code for Teacher: {self.teacher_id}"

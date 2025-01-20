from sqlalchemy import Column, String, ForeignKey, event
from sqlalchemy.orm import Mapped, relationship, validates
import re

from ..base import Base, TimestampMixin, SoftDeleteMixin


class User(Base, TimestampMixin, SoftDeleteMixin):
    """User model for all system users (students, teachers, admins)"""

    # Core fields
    email: Mapped[str] = Column(String(255), unique=True, nullable=False, index=True)
    role_name: Mapped[str] = Column(
        String(50), ForeignKey("role.role_name"), nullable=False
    )

    # Relationships
    role = relationship("Role", back_populates="users")
    created_texts = relationship(
        "Text", back_populates="teacher", foreign_keys="[Text.teacher_id]"
    )
    active_assessments = relationship(
        "ActiveAssessment",
        back_populates="student",
        foreign_keys="[ActiveAssessment.student_id]",
    )
    completions = relationship(
        "Completion", back_populates="student", foreign_keys="[Completion.student_id]"
    )

    @validates("email")
    def validate_email(self, key, email):
        """Validate email format"""
        if not email:
            raise ValueError("Email cannot be empty")

        # Basic email format validation
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")

        return email.lower()  # Store emails in lowercase

    @validates("role_name")
    def validate_role(self, key, role_name):
        """Validate role assignment"""
        valid_roles = {"STUDENT", "TEACHER", "ADMIN"}
        if role_name not in valid_roles:
            raise ValueError(f"Invalid role. Must be one of: {', '.join(valid_roles)}")
        return role_name

    def __str__(self) -> str:
        return f"{self.email} ({self.role_name})"


# SQLAlchemy event listeners for additional validation/processing
@event.listens_for(User, "before_insert")
def lowercase_email_insert(mapper, connection, target):
    """Ensure email is lowercase before insert"""
    if target.email:
        target.email = target.email.lower()


@event.listens_for(User, "before_update")
def lowercase_email_update(mapper, connection, target):
    """Ensure email is lowercase before update"""
    if target.email:
        target.email = target.email.lower()

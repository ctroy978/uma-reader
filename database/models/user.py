from sqlalchemy import Column, String, ForeignKey, event
from sqlalchemy.orm import Mapped, relationship, validates
import re

from ..base import Base, TimestampMixin, SoftDeleteMixin


class User(Base, TimestampMixin, SoftDeleteMixin):
    """User model for all system users (students, teachers, admins)"""

    # Core fields
    email: Mapped[str] = Column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = Column(String(50), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = Column(String(100), nullable=False)
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
    bypass_code = relationship(
        "TeacherBypassCode",
        back_populates="teacher",
        uselist=False,
        cascade="all, delete-orphan",
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

    @validates("username")
    def validate_username(self, key, username):
        """Validate username format"""
        if not username:
            raise ValueError("Username cannot be empty")

        # Username format validation: letters, numbers, underscore, 3-50 chars
        pattern = r"^[a-zA-Z0-9_]{3,50}$"
        if not re.match(pattern, username):
            raise ValueError(
                "Username must be 3-50 characters and contain only letters, numbers, and underscores"
            )

        return username.lower()  # Store usernames in lowercase

    @validates("full_name")
    def validate_full_name(self, key, full_name):
        """Validate full name"""
        if not full_name:
            raise ValueError("Full name cannot be empty")

        if len(full_name) > 100:
            raise ValueError("Full name must be less than 100 characters")

        # Remove excessive whitespace and capitalize properly
        return " ".join(full_name.split())

    @validates("role_name")
    def validate_role(self, key, role_name):
        """Validate role assignment"""
        valid_roles = {"STUDENT", "TEACHER", "ADMIN"}
        if role_name not in valid_roles:
            raise ValueError(f"Invalid role. Must be one of: {', '.join(valid_roles)}")
        return role_name

    def __str__(self) -> str:
        return f"{self.username} ({self.email}) - {self.role_name}"


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


@event.listens_for(User, "before_insert")
def lowercase_username_insert(mapper, connection, target):
    """Ensure username is lowercase before insert"""
    if target.username:
        target.username = target.username.lower()


@event.listens_for(User, "before_update")
def lowercase_username_update(mapper, connection, target):
    """Ensure username is lowercase before update"""
    if target.username:
        target.username = target.username.lower()

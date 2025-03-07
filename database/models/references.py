from sqlalchemy import Column, String, Integer, ForeignKey, Table
from sqlalchemy.orm import Mapped, relationship

from ..base import Base, TimestampMixin


class Role(Base, TimestampMixin):
    """Reference table for user roles"""

    role_name: Mapped[str] = Column(String(50), unique=True, nullable=False)
    description: Mapped[str] = Column(String(255))

    # Relationship
    users = relationship("User", back_populates="role")

    def __str__(self) -> str:
        return self.role_name


class TextForm(Base, TimestampMixin):
    """Reference table for text forms (PROSE, POETRY, etc.)"""

    form_name: Mapped[str] = Column(String(50), unique=True, nullable=False)
    description: Mapped[str] = Column(String(255))

    # Relationship
    texts = relationship("Text", back_populates="form")

    def __str__(self) -> str:
        return self.form_name


class PrimaryType(Base, TimestampMixin):
    """Reference table for primary text types (NARRATIVE, INFORMATIONAL, etc.)"""

    type_name: Mapped[str] = Column(String(50), unique=True, nullable=False)
    description: Mapped[str] = Column(String(255))

    # Relationship
    texts = relationship("Text", back_populates="primary_type")

    def __str__(self) -> str:
        return self.type_name


# app/database/models/references.py - partial file, only showing the updated class
class QuestionCategory(Base, TimestampMixin):
    """Reference table for question categories with progression order"""

    category_name: Mapped[str] = Column(String(50), unique=True, nullable=False)
    description: Mapped[str] = Column(String(255))
    progression_order: Mapped[int] = Column(Integer, nullable=False, unique=True)

    # Relationships
    active_assessments = relationship(
        "ActiveAssessment", back_populates="current_category_ref"
    )
    completion_questions = relationship(
        "CompletionQuestion", back_populates="category_ref"
    )

    cached_questions = relationship("QuestionCache", back_populates="category")

    def __str__(self) -> str:
        return self.category_name


class QuestionDifficulty(Base, TimestampMixin):
    """Reference table for question difficulty levels"""

    difficulty_name: Mapped[str] = Column(String(50), unique=True, nullable=False)
    description: Mapped[str] = Column(String(255))
    level_value: Mapped[int] = Column(Integer, nullable=False, unique=True)

    # Relationships
    active_assessments = relationship(
        "ActiveAssessment",
        back_populates="current_difficulty_ref",  # Changed from current_difficulty
    )
    completion_questions = relationship(
        "CompletionQuestion", back_populates="difficulty_ref"
    )

    def __str__(self) -> str:
        return self.difficulty_name


class Genre(Base, TimestampMixin):
    """Reference table for text genres"""

    genre_name: Mapped[str] = Column(String(50), unique=True, nullable=False)
    description: Mapped[str] = Column(String(255))

    # Many-to-many relationship with texts
    texts = relationship("Text", secondary="text_genres", back_populates="genres")

    def __str__(self) -> str:
        return self.genre_name


# Junction table for Text-Genre many-to-many relationship
text_genres = Table(
    "text_genres",
    Base.metadata,
    Column(
        "text_id",
        String(36),
        ForeignKey("text.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("genre_name", String(50), ForeignKey("genre.genre_name"), primary_key=True),
)

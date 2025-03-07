from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    ForeignKey,
    CheckConstraint,
    event,
    select,
)
from sqlalchemy.orm import Mapped, relationship, validates

from ..base import Base, TimestampMixin, SoftDeleteMixin
from .references import text_genres  # Junction table


class Text(Base, TimestampMixin, SoftDeleteMixin):
    """Text model for reading materials (metadata only)"""

    # Core fields
    teacher_id: Mapped[str] = Column(
        String(36), ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    title: Mapped[str] = Column(String(255), nullable=False)
    grade_level: Mapped[int] = Column(Integer, nullable=False)
    form_name: Mapped[str] = Column(
        String(50), ForeignKey("textform.form_name"), nullable=False
    )
    type_name: Mapped[str] = Column(
        String(50), ForeignKey("primarytype.type_name"), nullable=False
    )
    avg_unit_length: Mapped[str] = Column(String(10), nullable=False)

    # Relationships
    teacher = relationship("User", back_populates="created_texts")
    form = relationship("TextForm", back_populates="texts")
    primary_type = relationship("PrimaryType", back_populates="texts")
    genres = relationship("Genre", secondary=text_genres, back_populates="texts")
    chunks = relationship(
        "Chunk",
        back_populates="text",
        cascade="all, delete-orphan",
        order_by="Chunk.created_at",
    )
    active_assessments = relationship("ActiveAssessment", back_populates="text")
    completions = relationship("Completion", back_populates="text")

    # Constraints
    __table_args__ = (
        CheckConstraint("grade_level BETWEEN 2 AND 12", name="valid_grade_level"),
        CheckConstraint(
            "avg_unit_length IN ('SHORT', 'MEDIUM', 'LONG')", name="valid_unit_length"
        ),
    )

    @validates("avg_unit_length")
    def validate_unit_length(self, key, value):
        """Validate unit length value"""
        valid_lengths = {"SHORT", "MEDIUM", "LONG"}
        if value not in valid_lengths:
            raise ValueError(
                f"Invalid unit length. Must be one of: {', '.join(valid_lengths)}"
            )
        return value

    @validates("grade_level")
    def validate_grade_level(self, key, value):
        """Validate grade level range"""
        if not 2 <= value <= 12:
            raise ValueError("Grade level must be between 2 and 12")
        return value

    def __str__(self) -> str:
        return f"{self.title} (Grade {self.grade_level})"


class Chunk(Base, TimestampMixin, SoftDeleteMixin):
    """Chunk model for actual text content in chainable format"""

    # Core fields
    text_id: Mapped[str] = Column(
        String(36), ForeignKey("text.id", ondelete="CASCADE"), nullable=False
    )
    next_chunk_id: Mapped[str] = Column(
        String(36), ForeignKey("chunk.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = Column(String, nullable=False)
    is_first: Mapped[bool] = Column(Boolean, nullable=False, default=False)
    word_count: Mapped[int] = Column(Integer, nullable=False)

    # Relationships
    text = relationship("Text", back_populates="chunks")
    next_chunk = relationship(
        "Chunk",
        remote_side="[Chunk.id]",  # Use string reference
        uselist=False,
        backref="previous_chunk",
        post_update=True,
    )
    active_assessments = relationship(
        "ActiveAssessment", back_populates="current_chunk"
    )
    # New relationship for simplified versions - use string to avoid circular import
    simplified_versions = relationship(
        "SimplifiedChunk", back_populates="chunk", cascade="all, delete-orphan"
    )

    cached_questions = relationship(
        "QuestionCache", back_populates="chunk", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (CheckConstraint("word_count > 0", name="valid_word_count"),)

    @validates("word_count")
    def validate_word_count(self, key, value):
        """Validate word count"""
        if value <= 0:
            raise ValueError("Word count must be greater than 0")
        return value

    def __str__(self) -> str:
        return f"Chunk {self.id} of {self.text_id}"


# In database/models/text.py - replace or update the existing event listeners


@event.listens_for(Chunk, "before_insert")
def validate_chunk_chain(mapper, connection, target):
    """Prevent circular references and multiple incoming links on insert"""
    # Skip validation during insert since IDs might not be assigned yet
    pass


@event.listens_for(Chunk, "before_update")
def validate_chunk_updates(mapper, connection, target):
    """Validate chunk updates maintain chain integrity"""
    if target.next_chunk_id is not None:
        # Check for self-reference first (simpler case)
        if target.next_chunk_id == target.id:
            raise ValueError("Chunk cannot reference itself as next chunk")

        # Check for circular references in the chain
        seen_chunks = {target.id}
        current_id = target.next_chunk_id

        while current_id is not None:
            if current_id in seen_chunks:
                raise ValueError("Circular reference detected in chunk chain")
            seen_chunks.add(current_id)

            # Get the next chunk's next_chunk_id
            stmt = select(Chunk.next_chunk_id).where(Chunk.id == current_id)
            result = connection.execute(stmt).first()
            if result is None:
                break
            current_id = result[0]

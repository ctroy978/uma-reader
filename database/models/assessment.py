# app/database/models/assessment.py
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    ForeignKey,
    CheckConstraint,
    event,
    DateTime,
)
from sqlalchemy.orm import Mapped, relationship, validates

from ..base import Base, TimestampMixin


class ActiveAssessment(Base, TimestampMixin):
    """Tracks active reading assessment sessions"""

    # Core fields
    student_id: Mapped[str] = Column(String(36), ForeignKey("user.id"), nullable=False)
    text_id: Mapped[str] = Column(String(36), ForeignKey("text.id"), nullable=False)
    current_chunk_id: Mapped[str] = Column(
        String(36),
        ForeignKey("chunk.id"),
        nullable=True,  # Null when assessment is complete
    )

    # Assessment state
    started_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    last_activity: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    current_category: Mapped[str] = Column(
        String(50), ForeignKey("questioncategory.category_name"), nullable=False
    )
    current_difficulty: Mapped[str] = Column(
        String(50), ForeignKey("questiondifficulty.difficulty_name"), nullable=False
    )

    # Progress tracking
    consecutive_correct: Mapped[int] = Column(Integer, nullable=False, default=0)
    consecutive_incorrect: Mapped[int] = Column(Integer, nullable=False, default=0)

    # Success rates by category (0-100)
    literal_basic_success: Mapped[float] = Column(Float, nullable=False, default=0.0)
    literal_detailed_success: Mapped[float] = Column(Float, nullable=False, default=0.0)
    vocabulary_success: Mapped[float] = Column(Float, nullable=False, default=0.0)
    inferential_simple_success: Mapped[float] = Column(
        Float, nullable=False, default=0.0
    )
    inferential_complex_success: Mapped[float] = Column(
        Float, nullable=False, default=0.0
    )
    structural_basic_success: Mapped[float] = Column(Float, nullable=False, default=0.0)
    structural_advanced_success: Mapped[float] = Column(
        Float, nullable=False, default=0.0
    )

    # Attempt counts by category
    literal_basic_attempts: Mapped[int] = Column(Integer, nullable=False, default=0)
    literal_detailed_attempts: Mapped[int] = Column(Integer, nullable=False, default=0)
    vocabulary_attempts: Mapped[int] = Column(Integer, nullable=False, default=0)
    inferential_simple_attempts: Mapped[int] = Column(
        Integer, nullable=False, default=0
    )
    inferential_complex_attempts: Mapped[int] = Column(
        Integer, nullable=False, default=0
    )
    structural_basic_attempts: Mapped[int] = Column(Integer, nullable=False, default=0)
    structural_advanced_attempts: Mapped[int] = Column(
        Integer, nullable=False, default=0
    )

    # Status flags
    is_active: Mapped[bool] = Column(Boolean, nullable=False, default=True)
    completed: Mapped[bool] = Column(Boolean, nullable=False, default=False)

    # Relationships
    student = relationship("User", back_populates="active_assessments")
    text = relationship("Text", back_populates="active_assessments")
    current_chunk = relationship("Chunk", back_populates="active_assessments")
    current_category_ref = relationship(
        "QuestionCategory", back_populates="active_assessments"
    )
    current_difficulty_ref = relationship(
        "QuestionDifficulty", back_populates="active_assessments"
    )
    completions = relationship("Completion", back_populates="assessment")

    # Constraints
    __table_args__ = (
        CheckConstraint("consecutive_correct >= 0", name="valid_consecutive_correct"),
        CheckConstraint(
            "consecutive_incorrect >= 0", name="valid_consecutive_incorrect"
        ),
        CheckConstraint(
            "literal_basic_success BETWEEN 0 AND 100",
            name="valid_literal_basic_success",
        ),
        CheckConstraint(
            "literal_detailed_success BETWEEN 0 AND 100",
            name="valid_literal_detailed_success",
        ),
        CheckConstraint(
            "vocabulary_success BETWEEN 0 AND 100", name="valid_vocabulary_success"
        ),
        CheckConstraint(
            "inferential_simple_success BETWEEN 0 AND 100",
            name="valid_inferential_simple_success",
        ),
        CheckConstraint(
            "inferential_complex_success BETWEEN 0 AND 100",
            name="valid_inferential_complex_success",
        ),
        CheckConstraint(
            "structural_basic_success BETWEEN 0 AND 100",
            name="valid_structural_basic_success",
        ),
        CheckConstraint(
            "structural_advanced_success BETWEEN 0 AND 100",
            name="valid_structural_advanced_success",
        ),
    )

    @validates("consecutive_correct", "consecutive_incorrect")
    def validate_consecutive_counts(self, key, value):
        """Validate consecutive counts are non-negative"""
        if value < 0:
            raise ValueError(f"{key} cannot be negative")
        return value

    @validates(
        "literal_basic_success",
        "literal_detailed_success",
        "vocabulary_success",
        "inferential_simple_success",
        "inferential_complex_success",
        "structural_basic_success",
        "structural_advanced_success",
    )
    def validate_success_rate(self, key, value):
        """Validate success rates are between 0 and 100"""
        if not 0 <= value <= 100:
            raise ValueError(f"{key} must be between 0 and 100")
        return value

    def __str__(self) -> str:
        return f"Assessment {self.id} - Student {self.student_id} - Text {self.text_id}"

    def update_success_rate(self, category: str, is_correct: bool) -> None:
        """Update success rate and attempt count for a given category"""
        attempts_field = f"{category}_attempts"
        success_field = f"{category}_success"

        current_attempts = getattr(self, attempts_field)
        current_success = getattr(self, success_field)

        # Increment attempts
        setattr(self, attempts_field, current_attempts + 1)

        # Update success rate
        new_success = (
            (current_success * current_attempts) + (100 if is_correct else 0)
        ) / (current_attempts + 1)
        setattr(self, success_field, new_success)

        # Update consecutive counters
        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
        else:
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0

        # Update last activity timestamp
        self.last_activity = datetime.now(timezone.utc)


@event.listens_for(ActiveAssessment, "before_insert")
def set_initial_category(mapper, connection, target):
    """Set initial category and difficulty if not specified"""
    if not target.current_category:
        target.current_category = "literal_basic"
    if not target.current_difficulty:
        target.current_difficulty = "basic"


@event.listens_for(ActiveAssessment, "before_update")
def validate_completion_state(mapper, connection, target):
    """Ensure completed assessments cannot be modified"""
    if target.completed and target.is_active:
        raise ValueError("Completed assessments cannot be active")

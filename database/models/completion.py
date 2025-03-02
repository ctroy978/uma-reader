# app/database/models/completion.py
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
from sqlalchemy import Text as SQLText  # Import as SQLText to avoid conflict

from sqlalchemy.orm import Mapped, relationship, validates

from ..base import Base, TimestampMixin, SoftDeleteMixin


class Completion(Base, TimestampMixin, SoftDeleteMixin):
    """Records completed reading assessments and final results"""

    # Core fields
    student_id: Mapped[str] = Column(String(36), ForeignKey("user.id"), nullable=False)
    text_id: Mapped[str] = Column(String(36), ForeignKey("text.id"), nullable=False)
    assessment_id: Mapped[str] = Column(
        String(36), ForeignKey("activeassessment.id"), nullable=False
    )

    # Test state tracking
    test_status: Mapped[str] = Column(
        String(20), nullable=False, default="pending", server_default="pending"
    )

    # Timestamps for test progression
    completion_triggered_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    test_started_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=True)

    # Final assessment state
    final_test_level: Mapped[str] = Column(
        String(50), ForeignKey("questioncategory.category_name"), nullable=False
    )
    final_test_difficulty: Mapped[str] = Column(
        String(50), ForeignKey("questiondifficulty.difficulty_name"), nullable=False
    )

    # Performance metrics
    overall_score: Mapped[float] = Column(Float, nullable=False, default=0.0)
    total_questions: Mapped[int] = Column(Integer, nullable=False, default=0)
    correct_answers: Mapped[int] = Column(Integer, nullable=False, default=0)

    # ai analysis
    analysis_content: Mapped[str] = Column(SQLText, nullable=True)

    # Category success rates (0-100)
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

    # Review flag
    needs_review: Mapped[bool] = Column(Boolean, nullable=False, default=False)

    # Relationships
    student = relationship("User", back_populates="completions")
    text = relationship("Text", back_populates="completions")
    assessment = relationship("ActiveAssessment", back_populates="completions")
    questions = relationship(
        "CompletionQuestion", back_populates="completion", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("overall_score BETWEEN 0 AND 100", name="valid_overall_score"),
        CheckConstraint("total_questions >= 0", name="valid_total_questions"),
        CheckConstraint("correct_answers >= 0", name="valid_correct_answers"),
        CheckConstraint(
            "correct_answers <= total_questions", name="valid_answer_count"
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

    @validates("test_status")
    def validate_test_status(self, key, value):
        """Validate test status is one of the allowed values"""
        allowed_statuses = {"pending", "in_progress", "completed"}
        if value not in allowed_statuses:
            raise ValueError(
                f"Invalid test status. Must be one of: {', '.join(allowed_statuses)}"
            )
        return value

    def start_test(self) -> None:
        """Mark the test as started"""
        if self.test_status != "pending":
            raise ValueError("Can only start tests that are pending")
        self.test_status = "in_progress"
        self.test_started_at = datetime.now(timezone.utc)

    def complete_test(self) -> None:
        """Mark the test as completed and calculate final scores"""
        if self.test_status != "in_progress":
            raise ValueError("Can only complete tests that are in progress")
        self.test_status = "completed"
        self.completed_at = datetime.now(timezone.utc)
        self.calculate_overall_score()

    def calculate_overall_score(self) -> None:
        """Calculate overall score from category success rates"""
        if self.total_questions > 0:
            self.overall_score = (self.correct_answers * 100) / self.total_questions
        else:
            self.overall_score = 0.0

    def __str__(self) -> str:
        return f"Completion {self.id} - Status: {self.test_status}"


class CompletionQuestion(Base, TimestampMixin):
    """Records individual questions and answers from the completion assessment"""

    # Core fields
    completion_id: Mapped[str] = Column(
        String(36), ForeignKey("completion.id", ondelete="CASCADE"), nullable=False
    )
    next_question_id: Mapped[str] = Column(
        String(36), ForeignKey("completionquestion.id"), nullable=True
    )

    # Question details
    category: Mapped[str] = Column(
        String(50), ForeignKey("questioncategory.category_name"), nullable=False
    )
    difficulty: Mapped[str] = Column(
        String(50), ForeignKey("questiondifficulty.difficulty_name"), nullable=False
    )
    question_text: Mapped[str] = Column(String, nullable=False)
    student_answer: Mapped[str] = Column(String, nullable=True)
    is_correct: Mapped[bool] = Column(Boolean, nullable=True)
    is_answered: Mapped[bool] = Column(Boolean, nullable=False, default=False)
    time_spent_seconds: Mapped[int] = Column(Integer, nullable=True)

    # Relationships
    completion = relationship("Completion", back_populates="questions")
    next_question = relationship(
        "CompletionQuestion",
        remote_side="[CompletionQuestion.id]",
        uselist=False,
        backref="previous_question",
        post_update=True,
    )
    category_ref = relationship(
        "QuestionCategory", back_populates="completion_questions"
    )
    difficulty_ref = relationship(
        "QuestionDifficulty", back_populates="completion_questions"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("time_spent_seconds >= 0", name="valid_time_spent"),
    )

    def __str__(self) -> str:
        status = "Answered" if self.is_answered else "Unanswered"
        return f"Question {self.id} - {status}"

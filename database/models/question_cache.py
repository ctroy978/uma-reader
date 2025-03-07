# app/database/models/question_cache.py
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, ForeignKey, Text, DateTime, Index
from sqlalchemy.orm import Mapped, relationship

from ..base import Base, TimestampMixin, SoftDeleteMixin


class QuestionCache(Base, TimestampMixin, SoftDeleteMixin):
    """Caches AI-generated questions for text chunks by category"""

    # Core fields
    id: Mapped[str] = Column(String(36), primary_key=True)
    chunk_id: Mapped[str] = Column(String(36), ForeignKey("chunk.id"), nullable=False)
    question_category: Mapped[str] = Column(
        String(50), ForeignKey("questioncategory.category_name"), nullable=False
    )
    grade_level: Mapped[int] = Column(Integer, nullable=False)
    question_text: Mapped[str] = Column(Text, nullable=False)

    # Usage tracking
    access_count: Mapped[int] = Column(Integer, nullable=False, default=0)
    last_accessed: Mapped[datetime] = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    chunk = relationship("Chunk", back_populates="cached_questions")
    category = relationship("QuestionCategory", back_populates="cached_questions")

    # Indexes for efficient lookup
    __table_args__ = (
        Index(
            "ix_question_cache_lookup", "chunk_id", "question_category", "grade_level"
        ),
    )

    def increment_access(self):
        """Increment access counter and update last_accessed timestamp"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def __str__(self) -> str:
        return f"Cached question for chunk {self.chunk_id}, category {self.question_category}"

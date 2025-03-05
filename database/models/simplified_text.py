from sqlalchemy import Column, String, Integer, ForeignKey, Text as SQLText, Index
from sqlalchemy.orm import Mapped, relationship

from ..base import Base, TimestampMixin, SoftDeleteMixin


class SimplifiedChunk(Base, TimestampMixin, SoftDeleteMixin):
    """Model for storing cached simplified versions of text chunks"""

    # Core fields
    chunk_id: Mapped[str] = Column(
        String(36), ForeignKey("chunk.id", ondelete="CASCADE"), nullable=False
    )
    original_grade_level: Mapped[int] = Column(Integer, nullable=False)
    target_grade_level: Mapped[int] = Column(Integer, nullable=False)
    simplified_content: Mapped[str] = Column(SQLText, nullable=False)

    # Usage statistics
    access_count: Mapped[int] = Column(Integer, default=0, nullable=False)

    # Relationship to the original chunk (reference by string to avoid circular imports)
    chunk = relationship("Chunk", back_populates="simplified_versions")

    # Indexes for efficient querying
    __table_args__ = (
        Index(
            "ix_simplified_chunk_chunk_target",
            "chunk_id",
            "target_grade_level",
            unique=True,
        ),
    )

    def __str__(self) -> str:
        return f"Simplified version of Chunk {self.chunk_id} (Grade {self.original_grade_level} -> {self.target_grade_level})"

    def increment_access_count(self):
        """Increment the access count for this simplified chunk"""
        self.access_count += 1

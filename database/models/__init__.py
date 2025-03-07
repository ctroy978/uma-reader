# app/database/models/__init__.py
from database.base import Base
from database.models.references import (
    Role,
    TextForm,
    PrimaryType,
    QuestionCategory,
    QuestionDifficulty,
    Genre,
)
from database.models.user import User
from database.models.text import Text, Chunk
from database.models.assessment import ActiveAssessment
from database.models.completion import Completion, CompletionQuestion
from .token import RefreshToken
from .question_cache import QuestionCache
from .whitelist import EmailWhitelist, WhitelistType


# Import SimplifiedChunk after all other models to avoid circular imports
from .simplified_text import SimplifiedChunk

# List all models for easy access
__all__ = [
    "Role",
    "TextForm",
    "PrimaryType",
    "QuestionCategory",
    "QuestionDifficulty",
    "Genre",
    "User",
    "Text",
    "Chunk",
    "ActiveAssessment",
    "Completion",
    "CompletionQuestion",
    "RefreshToken",
    "SimplifiedChunk",
    "QuestionCache",
    "EmailWhitelist",
    "WhitelistType",
]

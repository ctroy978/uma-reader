from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel, constr, confloat, Field

import re
from enum import Enum
from sqlalchemy.exc import SQLAlchemyError
import datetime

from database.session import get_db
from database.models import User, Text, Chunk, Genre
from auth.middleware import require_teacher


# Enums for text metadata - matching database constraints
class TextForm(str, Enum):
    PROSE = "PROSE"
    POETRY = "POETRY"
    DRAMA = "DRAMA"
    OTHER = "OTHER"


class PrimaryType(str, Enum):
    NARRATIVE = "NARRATIVE"
    INFORMATIONAL = "INFORMATIONAL"
    PERSUASIVE = "PERSUASIVE"
    OTHER = "OTHER"


class Genre(str, Enum):
    FANTASY = "FANTASY"
    MYTHOLOGY = "MYTHOLOGY"
    REALISTIC = "REALISTIC"
    HISTORICAL = "HISTORICAL"
    TECHNICAL = "TECHNICAL"
    BIOGRAPHY = "BIOGRAPHY"
    ADVENTURE = "ADVENTURE"
    MYSTERY = "MYSTERY"
    NONFICTION = "NONFICTION"
    OTHER = "OTHER"


class TextProcessor:
    """Process text content with XML chunks"""

    @staticmethod
    def extract_title_and_chunks(content: str) -> Tuple[str, List[str]]:
        """Extract title and chunks from XML content"""
        # Extract title
        title_match = re.search(r"<title>(.*?)</title>", content, re.DOTALL)
        if not title_match:
            raise ValueError("No title found in content")
        title = title_match.group(1).strip()

        # Extract chunks - using non-greedy match to handle multiple chunks
        chunks = re.findall(r"<chunk>(.*?)</chunk>", content, re.DOTALL)
        if not chunks:
            raise ValueError("No chunks found in content")

        # Clean and validate chunks
        cleaned_chunks = [chunk.strip() for chunk in chunks]
        if any(not chunk for chunk in cleaned_chunks):
            raise ValueError("Empty chunks are not allowed")

        return title, cleaned_chunks

    @staticmethod
    def calculate_word_count(text: str) -> int:
        """Calculate word count for a chunk"""
        # Split on whitespace and filter out empty strings
        words = [word for word in text.split() if word]
        return len(words)

    @staticmethod
    def determine_avg_unit_length(chunks: List[str]) -> str:
        """Determine average unit length based on chunk sizes"""
        avg_words = sum(
            TextProcessor.calculate_word_count(chunk) for chunk in chunks
        ) / len(chunks)
        if avg_words < 50:
            return "SHORT"
        elif avg_words < 200:
            return "MEDIUM"
        else:
            return "LONG"


# Pydantic models for request validation
class TextCreate(BaseModel):
    grade_level: int = Field(..., ge=2, le=12)
    form: TextForm
    primary_type: PrimaryType
    genres: Set[Genre]


class TextResponse(BaseModel):
    id: str
    title: str
    grade_level: int
    form: TextForm
    primary_type: PrimaryType
    genres: List[str]
    chunk_count: int
    avg_unit_length: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True, "arbitrary_types_allowed": True}


router = APIRouter(prefix="/texts", tags=["Texts"])


@router.post("/", response_model=TextResponse, status_code=status.HTTP_201_CREATED)
async def create_text(
    text_data: TextCreate,
    content: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher),
):
    """Create a new text with chunks from XML content"""
    try:
        # Process the XML content
        title, chunks = TextProcessor.extract_title_and_chunks(content)

        # Calculate average unit length
        avg_unit_length = TextProcessor.determine_avg_unit_length(chunks)

        # Start database transaction
        db_text = Text(
            teacher_id=current_user.id,
            title=title,
            grade_level=text_data.grade_level,
            form_name=text_data.form.value,
            type_name=text_data.primary_type.value,
            avg_unit_length=avg_unit_length,
        )

        db.add(db_text)
        db.flush()  # Get the text ID without committing

        # Create chunks with proper linking
        previous_chunk = None
        first_chunk = None

        for chunk_content in chunks:
            word_count = TextProcessor.calculate_word_count(chunk_content)

            current_chunk = Chunk(
                text_id=db_text.id,
                content=chunk_content,
                word_count=word_count,
                is_first=(previous_chunk is None),
            )

            db.add(current_chunk)
            db.flush()  # Get the chunk ID

            if previous_chunk:
                previous_chunk.next_chunk_id = current_chunk.id
            else:
                first_chunk = current_chunk

            previous_chunk = current_chunk

        # Add genres
        for genre_name in text_data.genres:
            genre = db.query(Genre).filter(Genre.genre_name == genre_name.value).first()
            if genre:
                db_text.genres.append(genre)

        db.commit()

        return TextResponse(
            id=db_text.id,
            title=title,
            grade_level=text_data.grade_level,
            form=text_data.form,
            primary_type=text_data.primary_type,
            genres=[genre.genre_name for genre in db_text.genres],
            chunk_count=len(chunks),
            avg_unit_length=avg_unit_length,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred",
        )


@router.get("/", response_model=List[TextResponse])
async def get_texts(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher),
):
    """Get all texts for the current teacher"""
    texts = (
        db.query(Text)
        .filter(Text.teacher_id == current_user.id, Text.is_deleted == False)
        .order_by(Text.created_at.desc())
        .all()
    )
    return texts

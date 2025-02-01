from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from typing import List, Optional, Set, Tuple
from pydantic import BaseModel, constr, confloat, Field

import re
from enum import Enum
from sqlalchemy.exc import SQLAlchemyError
import datetime
import json

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
    text_data: str = Form(...),  # Changed to receive form data as string
    content: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher),
):
    """Create a new text with chunks from XML content"""
    try:
        # Parse the text_data JSON string into a TextCreate model
        try:
            data = json.loads(text_data)
            text_create = TextCreate(
                grade_level=int(data["grade_level"]),
                form=data["form"],
                primary_type=data["primary_type"],
                genres=(
                    set(data["genres"]) if isinstance(data["genres"], list) else set()
                ),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid text metadata: {str(e)}",
            )

        # Process the XML content
        title, chunks = TextProcessor.extract_title_and_chunks(content)

        # Calculate average unit length
        avg_unit_length = TextProcessor.determine_avg_unit_length(chunks)

        # Start database transaction
        db_text = Text(
            teacher_id=current_user.id,
            title=title,
            grade_level=text_create.grade_level,
            form_name=text_create.form.value,
            type_name=text_create.primary_type.value,
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
        for genre_name in text_create.genres:
            genre = db.query(Genre).filter(Genre.genre_name == genre_name.value).first()
            if genre:
                db_text.genres.append(genre)

        db.commit()

        return TextResponse(
            id=db_text.id,
            title=title,
            grade_level=text_create.grade_level,
            form=text_create.form,
            primary_type=text_create.primary_type,
            genres=[genre.genre_name for genre in db_text.genres],
            chunk_count=len(chunks),
            avg_unit_length=avg_unit_length,
            created_at=db_text.created_at,
            updated_at=db_text.updated_at,
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

    # Transform the database models into response models
    responses = []
    for text in texts:
        # Count chunks
        chunk_count = (
            db.query(Chunk)
            .filter(Chunk.text_id == text.id, Chunk.is_deleted == False)
            .count()
        )

        responses.append(
            TextResponse(
                id=text.id,
                title=text.title,
                grade_level=text.grade_level,
                form=text.form_name,  # Map form_name to form
                primary_type=text.type_name,  # Map type_name to primary_type
                genres=[genre.genre_name for genre in text.genres],
                chunk_count=chunk_count,
                avg_unit_length=text.avg_unit_length,
                created_at=text.created_at,
                updated_at=text.updated_at,
            )
        )

    return responses


# Add this to texts.py


class TextDetailResponse(BaseModel):
    text: TextResponse
    chunks: List[dict]


@router.get("/{text_id}", response_model=TextDetailResponse)
async def get_text(
    text_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher),
):
    """Get a specific text with its chunks"""
    # Get the text
    text = (
        db.query(Text)
        .filter(
            Text.id == text_id,
            Text.teacher_id == current_user.id,
            Text.is_deleted == False,
        )
        .first()
    )

    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    # Get chunks in order
    chunks = []
    current_chunk = (
        db.query(Chunk)
        .filter(
            Chunk.text_id == text_id, Chunk.is_deleted == False, Chunk.is_first == True
        )
        .first()
    )

    while current_chunk:
        chunks.append(
            {
                "id": current_chunk.id,
                "content": current_chunk.content,
                "word_count": current_chunk.word_count,
            }
        )

        if not current_chunk.next_chunk_id:
            break

        current_chunk = (
            db.query(Chunk)
            .filter(Chunk.id == current_chunk.next_chunk_id, Chunk.is_deleted == False)
            .first()
        )

    # Count total chunks
    chunk_count = len(chunks)

    # Create response
    text_response = TextResponse(
        id=text.id,
        title=text.title,
        grade_level=text.grade_level,
        form=text.form_name,
        primary_type=text.type_name,
        genres=[genre.genre_name for genre in text.genres],
        chunk_count=chunk_count,
        avg_unit_length=text.avg_unit_length,
        created_at=text.created_at,
        updated_at=text.updated_at,
    )

    return TextDetailResponse(text=text_response, chunks=chunks)


from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

# Add to existing imports in texts.py
from database.models import ActiveAssessment


# New Pydantic models for responses
class ActiveAssessmentInfo(BaseModel):
    count: int
    student_names: List[str]

    class Config:
        from_attributes = True


# Add these new routes to the existing router


@router.get("/{text_id}/active-assessments", response_model=ActiveAssessmentInfo)
async def check_active_assessments(
    text_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher),
):
    """Check for active assessments on a text"""
    # Verify text exists and belongs to teacher
    text = (
        db.query(Text)
        .filter(
            Text.id == text_id,
            Text.teacher_id == current_user.id,
            Text.is_deleted == False,
        )
        .first()
    )

    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    # Get active assessments
    active_assessments = (
        db.query(ActiveAssessment)
        .filter(
            ActiveAssessment.text_id == text_id,
            ActiveAssessment.is_active == True,
            ActiveAssessment.completed == False,
        )
        .all()
    )

    # Get student names
    student_names = []
    for assessment in active_assessments:
        student = db.query(User).filter(User.id == assessment.student_id).first()
        if student:
            student_names.append(student.full_name)

    return ActiveAssessmentInfo(
        count=len(active_assessments), student_names=student_names
    )


class TextUpdateRequest(BaseModel):
    metadata: TextCreate
    content: str
    force: bool = False  # If True, will proceed even with active assessments


@router.put("/{text_id}", response_model=TextResponse)
async def update_text(
    text_id: str,
    update_data: TextUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_teacher),
):
    """Update a text and its chunks, handling active assessments"""
    # Start transaction
    try:
        # Verify text exists and belongs to teacher
        text = (
            db.query(Text)
            .filter(
                Text.id == text_id,
                Text.teacher_id == current_user.id,
                Text.is_deleted == False,
            )
            .first()
        )

        if not text:
            raise HTTPException(status_code=404, detail="Text not found")

        # Check for active assessments if not forced
        if not update_data.force:
            active_count = (
                db.query(ActiveAssessment)
                .filter(
                    ActiveAssessment.text_id == text_id,
                    ActiveAssessment.is_active == True,
                    ActiveAssessment.completed == False,
                )
                .count()
            )

            if active_count > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Text has {active_count} active assessments. Set force=true to proceed.",
                )

        # Process the new content
        try:
            title, chunks = TextProcessor.extract_title_and_chunks(update_data.content)
            avg_unit_length = TextProcessor.determine_avg_unit_length(chunks)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Soft delete active assessments
        db.query(ActiveAssessment).filter(
            ActiveAssessment.text_id == text_id,
            ActiveAssessment.is_active == True,
            ActiveAssessment.completed == False,
        ).update({"is_active": False, "completed": True})

        # Soft delete existing chunks
        db.query(Chunk).filter(
            Chunk.text_id == text_id, Chunk.is_deleted == False
        ).update({"is_deleted": True})

        # Create new chunks with proper linking
        previous_chunk = None
        first_chunk = None

        for chunk_content in chunks:
            word_count = TextProcessor.calculate_word_count(chunk_content)

            current_chunk = Chunk(
                text_id=text_id,
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

        # Update text metadata
        text.title = title
        text.grade_level = update_data.metadata.grade_level
        text.form_name = update_data.metadata.form.value
        text.type_name = update_data.metadata.primary_type.value
        text.avg_unit_length = avg_unit_length

        # Update genres
        text.genres = []  # Clear existing genres
        for genre_name in update_data.metadata.genres:
            genre = db.query(Genre).filter(Genre.genre_name == genre_name.value).first()
            if genre:
                text.genres.append(genre)

        db.commit()

        return TextResponse(
            id=text.id,
            title=title,
            grade_level=update_data.metadata.grade_level,
            form=update_data.metadata.form,
            primary_type=update_data.metadata.primary_type,
            genres=[genre.genre_name for genre in text.genres],
            chunk_count=len(chunks),
            avg_unit_length=avg_unit_length,
            created_at=text.created_at,
            updated_at=text.updated_at,
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

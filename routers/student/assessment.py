# app/api/endpoints/assessment.py

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from database.session import get_db
from database.models import Text, Chunk, ActiveAssessment, User
from auth.middleware import require_user

router = APIRouter(tags=["Assessment"])


# In assessment.py


class ChunkResponse(BaseModel):
    id: str
    content: str
    is_first: bool
    has_next: bool

    class Config:
        from_attributes = True


class StartAssessmentResponse(BaseModel):
    assessment_id: str
    text_title: str  # Add text title to the response
    chunk: ChunkResponse

    class Config:
        from_attributes = True


@router.post("/start/{text_id}", response_model=StartAssessmentResponse)
async def start_assessment(
    text_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Start a new assessment for a text and return the first chunk"""

    # Verify text exists and isn't deleted
    text = db.query(Text).filter(Text.id == text_id, Text.is_deleted == False).first()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Text not found"
        )

    # Check for existing active assessment
    existing_assessment = (
        db.query(ActiveAssessment)
        .filter(
            ActiveAssessment.student_id == user.id,
            ActiveAssessment.text_id == text_id,
            ActiveAssessment.is_active == True,
            ActiveAssessment.completed == False,
        )
        .first()
    )

    if existing_assessment:
        # Update last activity timestamp
        existing_assessment.last_activity = datetime.now(timezone.utc)
        db.commit()

        # Get current chunk for existing assessment
        current_chunk = db.query(Chunk).get(existing_assessment.current_chunk_id)
        if current_chunk:
            return StartAssessmentResponse(
                assessment_id=existing_assessment.id,
                text_title=text.title,  # Include text title
                chunk=ChunkResponse(
                    id=current_chunk.id,
                    content=current_chunk.content,
                    is_first=current_chunk.is_first,
                    has_next=current_chunk.next_chunk_id is not None,
                ),
            )

    # If no existing assessment, get the first chunk
    first_chunk = (
        db.query(Chunk)
        .filter(
            Chunk.text_id == text_id, Chunk.is_first == True, Chunk.is_deleted == False
        )
        .first()
    )
    if not first_chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Text has no content"
        )

    # Create new assessment
    assessment = ActiveAssessment(
        student_id=user.id,
        text_id=text_id,
        current_chunk_id=first_chunk.id,
        started_at=datetime.now(timezone.utc),
        last_activity=datetime.now(timezone.utc),
    )

    db.add(assessment)
    db.commit()

    # Return assessment ID, text title, and first chunk
    return StartAssessmentResponse(
        assessment_id=assessment.id,
        text_title=text.title,  # Include text title
        chunk=ChunkResponse(
            id=first_chunk.id,
            content=first_chunk.content,
            is_first=True,
            has_next=first_chunk.next_chunk_id is not None,
        ),
    )


@router.get("/next/{assessment_id}", response_model=ChunkResponse)
async def get_next_chunk(
    assessment_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get next chunk in reading sequence"""

    # Get active assessment
    assessment = (
        db.query(ActiveAssessment)
        .filter(
            ActiveAssessment.id == assessment_id,
            ActiveAssessment.student_id == user.id,
            ActiveAssessment.is_active == True,
            ActiveAssessment.completed == False,
        )
        .first()
    )

    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Active assessment not found"
        )

    # Get current chunk
    current_chunk = db.query(Chunk).get(assessment.current_chunk_id)
    if not current_chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Current chunk not found"
        )

    # Get next chunk
    if not current_chunk.next_chunk_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No next chunk available"
        )

    next_chunk = db.query(Chunk).get(current_chunk.next_chunk_id)
    if not next_chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Next chunk not found"
        )

    # Update assessment's current chunk
    assessment.current_chunk_id = next_chunk.id
    assessment.last_activity = datetime.now(timezone.utc)
    db.commit()

    return ChunkResponse(
        id=next_chunk.id,
        content=next_chunk.content,
        is_first=next_chunk.is_first,
        has_next=next_chunk.next_chunk_id is not None,
    )

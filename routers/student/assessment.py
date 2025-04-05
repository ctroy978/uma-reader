# routers/student/assessment.py

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from database.session import get_db
from database.models import Text, Chunk, ActiveAssessment, User, Completion
from auth.middleware import require_user

router = APIRouter(tags=["Assessment"])


class ChunkResponse(BaseModel):
    id: str
    content: str
    is_first: bool
    has_next: bool

    class Config:
        from_attributes = True


class StartAssessmentResponse(BaseModel):
    assessment_id: str
    text_title: str
    chunk: ChunkResponse

    class Config:
        from_attributes = True


class CompletionResponse(BaseModel):
    """Response model for assessment completion"""

    message: str
    completion_id: str
    assessment_id: str
    text_title: str
    days_remaining: int  # Time remaining to take completion test
    completion_triggered_at: datetime

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
                text_title=text.title,
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
        text_title=text.title,
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


@router.post("/{assessment_id}/complete", response_model=CompletionResponse)
async def complete_assessment(
    assessment_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Complete assessment and create completion test record"""

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

    # Get text information
    text = db.query(Text).get(assessment.text_id)
    if not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Text not found"
        )

    # Get current chunk
    current_chunk = db.query(Chunk).get(assessment.current_chunk_id)
    if not current_chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Current chunk not found"
        )

    # Verify this is the last chunk
    if current_chunk.next_chunk_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment is not at final chunk",
        )

    # Check for existing completion test
    existing_completion = (
        db.query(Completion)
        .filter(
            Completion.assessment_id == assessment_id, Completion.is_deleted == False
        )
        .first()
    )

    if existing_completion:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Completion test already exists",
        )

    # Start database transaction
    try:
        # 1. Mark assessment as complete
        assessment.completed = True
        assessment.is_active = False

        # 2. Create completion test record
        completion_triggered_at = datetime.now(timezone.utc)
        completion = Completion(
            student_id=user.id,
            text_id=assessment.text_id,
            assessment_id=assessment_id,
            final_test_level=assessment.current_category,
            final_test_difficulty=assessment.current_difficulty,
            completion_triggered_at=completion_triggered_at,
            test_status="pending",
        )

        db.add(completion)
        db.commit()

        # Calculate days remaining (14 days from trigger date)
        days_remaining = 14

        return CompletionResponse(
            message="Assessment completed successfully. A completion test has been created.",
            completion_id=completion.id,
            assessment_id=assessment_id,
            text_title=text.title,
            days_remaining=days_remaining,
            completion_triggered_at=completion_triggered_at,
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error completing assessment: {str(e)}",
        )


@router.get("/status/{text_id}", response_model=dict)
async def get_assessment_status(
    text_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Check if student has an active or completed assessment for this text"""

    # Check for existing active assessment
    active_assessment = (
        db.query(ActiveAssessment)
        .filter(
            ActiveAssessment.student_id == user.id,
            ActiveAssessment.text_id == text_id,
            ActiveAssessment.is_active == True,
            ActiveAssessment.completed == False,
        )
        .first()
    )

    # Check for completed assessment
    completed_assessment = (
        db.query(ActiveAssessment)
        .filter(
            ActiveAssessment.student_id == user.id,
            ActiveAssessment.text_id == text_id,
            ActiveAssessment.completed == True,
        )
        .first()
    )

    # Check for completion record if assessment is completed
    completion = None
    if completed_assessment:
        completion = (
            db.query(Completion)
            .filter(
                Completion.assessment_id == completed_assessment.id,
                Completion.is_deleted == False,
            )
            .first()
        )

    return {
        "has_active_assessment": active_assessment is not None,
        "assessment_id": active_assessment.id if active_assessment else None,
        "is_completed": completed_assessment is not None,
        "completion_id": completion.id if completion else None,
        "completion_status": completion.test_status if completion else None,
    }


@router.get("/text/{text_id}/chunks", response_model=dict)
async def get_text_chunk_count(
    text_id: str, db: Session = Depends(get_db), user: User = Depends(require_user)
):
    """Get the total number of chunks for a text"""

    # Verify text exists
    text = db.query(Text).filter(Text.id == text_id, Text.is_deleted == False).first()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Text not found"
        )

    # Count chunks - either directly or by following the chunk chain
    total_chunks = (
        db.query(Chunk)
        .filter(Chunk.text_id == text_id, Chunk.is_deleted == False)
        .count()
    )

    return {"total_chunks": total_chunks}


@router.get("/text/{text_id}/chunk-position/{chunk_id}", response_model=dict)
async def get_chunk_position(
    text_id: str,
    chunk_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get the position of a chunk within its text"""

    # Verify text exists
    text = db.query(Text).filter(Text.id == text_id, Text.is_deleted == False).first()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Text not found"
        )

    # Verify chunk exists and belongs to the text
    target_chunk = (
        db.query(Chunk)
        .filter(
            Chunk.id == chunk_id, Chunk.text_id == text_id, Chunk.is_deleted == False
        )
        .first()
    )

    if not target_chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found"
        )

    # Find the first chunk of the text
    first_chunk = (
        db.query(Chunk)
        .filter(
            Chunk.text_id == text_id, Chunk.is_first == True, Chunk.is_deleted == False
        )
        .first()
    )

    if not first_chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Text structure is corrupted"
        )

    # Calculate position by following the chain from first chunk
    position = 1
    current_chunk = first_chunk

    while current_chunk and current_chunk.id != chunk_id:
        if not current_chunk.next_chunk_id:
            # We've reached the end without finding our chunk
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not in sequence"
            )

        current_chunk = db.query(Chunk).get(current_chunk.next_chunk_id)
        if current_chunk and not current_chunk.is_deleted:
            position += 1

    # Count total chunks for the text (reusing existing logic)
    total_chunks = (
        db.query(Chunk)
        .filter(Chunk.text_id == text_id, Chunk.is_deleted == False)
        .count()
    )

    return {"position": position, "total_chunks": total_chunks}

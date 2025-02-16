# routers/student/completion.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone, timedelta

from database.session import get_db
from database.models import Completion, ActiveAssessment, User
from auth.middleware import require_user

router = APIRouter(tags=["completion-tests"])


# Response Models
class CompletionTestInfo(BaseModel):
    id: str
    text_title: str
    triggered_at: datetime
    days_remaining: int
    test_status: str

    class Config:
        from_attributes = True


class AvailableCompletionsResponse(BaseModel):
    completions: List[CompletionTestInfo]


# Endpoints
@router.post("/assessments/{assessment_id}/trigger-completion")
async def trigger_completion_test(
    assessment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Create a completion test record when active assessment is finished"""

    # Get active assessment
    assessment = db.query(ActiveAssessment).get(assessment_id)
    if not assessment or assessment.student_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found"
        )

    # Verify assessment is complete
    if not assessment.completed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment is not completed",
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

    # Create new completion test
    completion = Completion(
        student_id=user.id,
        text_id=assessment.text_id,
        assessment_id=assessment_id,
        final_test_level=assessment.current_category,
        final_test_difficulty=assessment.current_difficulty,
        completion_triggered_at=datetime.now(timezone.utc),
    )

    db.add(completion)
    db.commit()
    db.refresh(completion)

    return {"id": completion.id, "status": "pending"}


@router.get("/completions/available", response_model=AvailableCompletionsResponse)
async def get_available_completions(
    db: Session = Depends(get_db), user: User = Depends(require_user)
):
    """Get all available completion tests for the student"""

    # Calculate cutoff date (14 days ago)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=14)

    # Get available completions
    completions = (
        db.query(Completion)
        .filter(
            Completion.student_id == user.id,
            Completion.test_status == "pending",
            Completion.is_deleted == False,
            Completion.completion_triggered_at >= cutoff_date,
        )
        .all()
    )

    # Format response with remaining days calculation
    result = []
    for completion in completions:
        days_remaining = (
            14 - (datetime.now(timezone.utc) - completion.completion_triggered_at).days
        )

        result.append(
            CompletionTestInfo(
                id=completion.id,
                text_title=completion.text.title,
                triggered_at=completion.completion_triggered_at,
                days_remaining=max(0, days_remaining),
                test_status=completion.test_status,
            )
        )

    return AvailableCompletionsResponse(completions=result)

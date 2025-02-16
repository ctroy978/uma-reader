# routers/student/completion.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone, timedelta

from database.session import get_db
from database.models import Completion, ActiveAssessment, User, Text
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

    class Config:
        from_attributes = True


@router.get("/completions/available", response_model=AvailableCompletionsResponse)
async def get_available_completions(
    db: Session = Depends(get_db), user: User = Depends(require_user)
):
    """Get all available completion tests for the student"""
    try:
        # Calculate cutoff date (14 days ago)
        current_time = datetime.now(timezone.utc)
        cutoff_date = current_time - timedelta(days=14)

        # Get available completions with their associated texts
        completions = (
            db.query(Completion, Text.title.label("text_title"))
            .join(Text, Completion.text_id == Text.id)
            .filter(
                Completion.student_id == user.id,
                Completion.test_status == "pending",
                Completion.is_deleted == False,
                Completion.completion_triggered_at >= cutoff_date,
            )
            .all()
        )

        # Format response
        completion_list = []
        for completion, text_title in completions:
            # Ensure triggered_at is timezone-aware
            triggered_at = completion.completion_triggered_at
            if triggered_at.tzinfo is None:
                triggered_at = triggered_at.replace(tzinfo=timezone.utc)

            # Calculate days remaining
            days_elapsed = (current_time - triggered_at).days
            days_remaining = max(0, 14 - days_elapsed)

            completion_list.append(
                CompletionTestInfo(
                    id=completion.id,
                    text_title=text_title,
                    triggered_at=triggered_at,
                    days_remaining=days_remaining,
                    test_status=completion.test_status,
                )
            )

        return AvailableCompletionsResponse(completions=completion_list)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching completion tests: {str(e)}",
        )


@router.post("/{completion_id}/start")
async def start_completion_test(
    completion_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Start a completion test"""
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.test_status == "pending",
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found or not available",
        )

    # Check if test is still within time limit
    current_time = datetime.now(timezone.utc)
    triggered_at = completion.completion_triggered_at
    if triggered_at.tzinfo is None:
        triggered_at = triggered_at.replace(tzinfo=timezone.utc)

    days_elapsed = (current_time - triggered_at).days
    if days_elapsed > 14:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Completion test has expired",
        )

    try:
        # Update completion status
        completion.test_status = "in_progress"
        completion.updated_at = current_time
        db.commit()

        return {
            "message": "Completion test started successfully",
            "completion_id": completion.id,
            "text_id": completion.text_id,
            "final_test_level": completion.final_test_level,
            "final_test_difficulty": completion.final_test_difficulty,
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting completion test: {str(e)}",
        )

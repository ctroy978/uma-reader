# routers/student/completion.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone, timedelta

from database.session import get_db
from database.models import Completion, ActiveAssessment, User, Text, CompletionQuestion
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
        # IMPORTANT: Now including both "pending" AND "in_progress" tests
        completions = (
            db.query(Completion, Text.title.label("text_title"))
            .join(Text, Completion.text_id == Text.id)
            .filter(
                Completion.student_id == user.id,
                Completion.test_status.in_(
                    ["pending", "in_progress"]
                ),  # Include in-progress tests
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
    """Start a completion test, resetting it if it was previously in progress"""
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
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
        # If the test was already in progress, reset it
        if completion.test_status == "in_progress":
            # Delete any existing questions
            questions = (
                db.query(CompletionQuestion)
                .filter(CompletionQuestion.completion_id == completion_id)
                .all()
            )

            for question in questions:
                db.delete(question)

            # Reset the completion status
            completion.test_status = "pending"
            completion.test_started_at = None
            completion.overall_score = 0.0
            completion.total_questions = 0
            completion.correct_answers = 0
            completion.literal_basic_success = 0.0
            completion.literal_detailed_success = 0.0
            completion.vocabulary_success = 0.0
            completion.inferential_simple_success = 0.0
            completion.inferential_complex_success = 0.0
            completion.structural_basic_success = 0.0
            completion.structural_advanced_success = 0.0

            # Log the reset
            print(f"Reset in-progress test {completion_id} for user {user.id}")

        # Update completion status to in_progress
        completion.test_status = "in_progress"
        completion.test_started_at = current_time
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


@router.get("/{completion_id}", response_model=dict)
async def get_completion_test(
    completion_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get details for a specific completion test"""
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found",
        )

    # Get text information
    text = db.query(Text).get(completion.text_id)
    if not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Text not found",
        )

    return {
        "completion_id": completion.id,
        "text_id": completion.text_id,
        "text_title": text.title,
        "assessment_id": completion.assessment_id,
        "test_status": completion.test_status,
        "final_test_level": completion.final_test_level,
        "final_test_difficulty": completion.final_test_difficulty,
        "days_remaining": 14,  # Calculate the actual days remaining here
    }

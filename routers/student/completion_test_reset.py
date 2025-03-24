# app/routers/student/completion_test_reset.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
from pydantic import BaseModel, field_validator
import re
from datetime import datetime, timezone
import logging

# Configure logger
logger = logging.getLogger(__name__)

from database.session import get_db
from database.models import Completion, TeacherBypassCode, User, ActiveAssessment
from auth.middleware import require_user

router = APIRouter(tags=["Completion Test"])


class BypassCodeRequest(BaseModel):
    """Schema for teacher bypass code verification"""

    bypass_code: str

    @field_validator("bypass_code")
    @classmethod
    def validate_bypass_code(cls, v):
        if not re.match(r"^\d{4}$", v):
            raise ValueError("Bypass code must be exactly 4 digits")
        return v


class ResetResponse(BaseModel):
    """Response schema for test reset operation"""

    message: str
    text_id: str
    new_assessment_id: str = ""  # Added field for new assessment ID


@router.post("/{completion_id}/reset-test", response_model=ResetResponse)
async def reset_test(
    completion_id: str,
    bypass_data: BypassCodeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """Reset a test after an integrity violation using teacher bypass code"""
    # Find the completion record
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == current_user.id,
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test not found or already completed",
        )

    # Store the text_id for the response
    text_id = completion.text_id
    assessment_id = completion.assessment_id

    # Verify the bypass code against any active teacher bypass code
    valid_bypass = (
        db.query(TeacherBypassCode)
        .filter(
            TeacherBypassCode.bypass_code == bypass_data.bypass_code,
            TeacherBypassCode.is_active == True,
        )
        .first()
    )

    if not valid_bypass:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bypass code. Please ask your teacher for assistance.",
        )

    try:
        logger.info(
            f"Resetting completion test {completion_id} for student {current_user.id}"
        )

        # 1. Reset the completion record to its initial state
        # Status - ensure it's set to "pending" to indicate a fresh test
        completion.test_status = "pending"

        # Timestamps - clear these to indicate test hasn't started/completed
        completion.test_started_at = None
        completion.completed_at = None

        # Reset all performance metrics
        completion.overall_score = 0.0
        completion.total_questions = 0
        completion.correct_answers = 0

        # Keep the final_test_level and final_test_difficulty (they come from the assessment)
        # No need to change these as they're determined by the reading assessment

        # Reset all category success rates
        completion.literal_basic_success = 0.0
        completion.literal_detailed_success = 0.0
        completion.vocabulary_success = 0.0
        completion.inferential_simple_success = 0.0
        completion.inferential_complex_success = 0.0
        completion.structural_basic_success = 0.0
        completion.structural_advanced_success = 0.0

        # Reset needs_review flag
        completion.needs_review = False

        # Update timestamp
        completion.updated_at = datetime.now(timezone.utc)

        logger.info(f"Reset completion {completion_id} to pending state")

        # 2. Delete all associated question records
        logger.info(
            f"Deleting existing question records for completion {completion_id}"
        )
        for question in completion.questions:
            db.delete(question)

        # 3. Ensure the associated assessment remains completed
        # Get the assessment associated with this completion
        assessment = (
            db.query(ActiveAssessment)
            .filter(
                ActiveAssessment.id == assessment_id,
            )
            .first()
        )

        if assessment:
            logger.info(
                f"Ensuring assessment {assessment_id} remains in completed state"
            )
            # Make sure the assessment stays completed (don't allow re-reading)
            assessment.is_active = False
            assessment.completed = True
            assessment.updated_at = datetime.now(timezone.utc)
        else:
            logger.warning(
                f"Assessment {assessment_id} not found for completion {completion_id}"
            )

        # 4. Commit all changes
        db.commit()
        logger.info(f"Successfully reset completion test {completion_id}")

        # Return success with the text_id for redirection
        return ResetResponse(
            message="Test has been reset successfully. You can now take it again from your dashboard.",
            text_id=text_id,
            new_assessment_id=assessment_id,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting test: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset test: {str(e)}",
        )

# app/routers/students/questions.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timezone

from database.session import get_db
from database.models import ActiveAssessment, User, QuestionCategory
from auth.middleware import require_user

router = APIRouter()  # Changed from questions_router to router


class QuestionResponse(BaseModel):
    category: str
    question_text: str
    assessment_id: str

    class Config:
        from_attributes = True


class AnswerRequest(BaseModel):
    assessment_id: str
    is_correct: bool


def get_dummy_question(category: str) -> str:
    """Generate a dummy question based on category."""
    return f"This is a {category} question about the text. What do you think about the key concepts presented here?"


def get_next_category(current: str) -> Optional[str]:
    """Get the next category in the progression."""
    progression = [
        "literal_basic",
        "literal_detailed",
        "inferential_simple",
        "inferential_complex",
    ]
    try:
        current_idx = progression.index(current)
        if current_idx < len(progression) - 1:
            return progression[current_idx + 1]
        return None
    except ValueError:
        return None


def get_previous_category(current: str) -> Optional[str]:
    """Get the previous category in the progression."""
    progression = [
        "literal_basic",
        "literal_detailed",
        "inferential_simple",
        "inferential_complex",
    ]
    try:
        current_idx = progression.index(current)
        if current_idx > 0:
            return progression[current_idx - 1]
        return None
    except ValueError:
        return None


@router.get(
    "/current/{assessment_id}", response_model=QuestionResponse
)  # Changed decorator to use router
async def get_current_question(
    assessment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get the current question for an assessment."""
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

    return QuestionResponse(
        category=assessment.current_category,
        question_text=get_dummy_question(assessment.current_category),
        assessment_id=assessment.id,
    )


@router.post(
    "/answer", response_model=QuestionResponse
)  # Changed decorator to use router
async def submit_answer(
    answer: AnswerRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Submit an answer and get the next question."""
    assessment = (
        db.query(ActiveAssessment)
        .filter(
            ActiveAssessment.id == answer.assessment_id,
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

    # Update success rate and consecutive counts
    assessment.update_success_rate(assessment.current_category, answer.is_correct)

    if answer.is_correct:
        # Check for promotion (2 consecutive correct)
        if assessment.consecutive_correct >= 2:
            next_category = get_next_category(assessment.current_category)
            if next_category:
                assessment.current_category = next_category
                assessment.consecutive_correct = 0
    else:
        # Check for demotion (2 consecutive incorrect)
        if assessment.consecutive_incorrect >= 2:
            prev_category = get_previous_category(assessment.current_category)
            if prev_category:
                assessment.current_category = prev_category
                assessment.consecutive_incorrect = 0

    # Update last activity
    assessment.last_activity = datetime.now(timezone.utc)
    db.commit()

    # Return the next question
    return QuestionResponse(
        category=assessment.current_category,
        question_text=get_dummy_question(assessment.current_category),
        assessment_id=assessment.id,
    )

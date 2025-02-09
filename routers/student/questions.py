# app/routers/students/questions.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel, Field
from database.session import get_db
from database.models import ActiveAssessment, User, QuestionCategory, Text, Chunk
from auth.middleware import require_user


router = APIRouter()  # Changed from questions_router to router

# Load environment variables and initialize AI model
load_dotenv()
model = GeminiModel("gemini-2.0-flash")


class Question(BaseModel):
    """Question model includes question, question_type, target_grade"""

    question: str = Field(..., description="The actual question text.")
    question_type: str = Field(..., description="the category of question")
    target_grade: int = Field(..., description="the grade level")


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


def build_prompt() -> str:
    return """
    You are an experienced reading teacher tasked with creating reading comprehension questions based on text chunks. Here's what you should do:

    - Analyze the text chunk provided in the query.
    - Generate a question matching the specified question_type and target_grade.
    - Ensure the question is clear, unambiguous, and fits the educational goals for the grade.
    - Maintain a supportive tone in your question.

    Use this format for your response:
    {"question": "Question text", "question_type": "category", "target_grade": grade_number}
    """


async def get_question(category: str, assessment_id: str, db: Session) -> str:
    """
    Generate a contextual question based on category and current text chunk.

    Args:
        category: The type of question to generate (literal_basic, inferential_simple, etc)
        assessment_id: The active assessment ID
        db: Database session

    Returns:
        str: Generated question text
    """
    # Get the assessment and related text/chunk
    assessment = (
        db.query(ActiveAssessment)
        .filter(
            ActiveAssessment.id == assessment_id,
            ActiveAssessment.is_active == True,
            ActiveAssessment.completed == False,
        )
        .first()
    )

    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Active assessment not found"
        )

    # Get current chunk and text metadata
    current_chunk = db.query(Chunk).get(assessment.current_chunk_id)
    text = db.query(Text).get(assessment.text_id)

    if not current_chunk or not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current chunk or text not found",
        )

    try:
        # Initialize AI agent
        agent = Agent(
            model=model,
            result_type=Question,
            system_prompt=build_prompt(),
        )

        # Construct query with text chunk, category, and grade level
        query = f"Generate a '{category}' question for grade {text.grade_level} based on this text: '{current_chunk.content}'"

        # Generate question using AI
        result = await agent.run(query)

        # Return the generated question text
        return result.data.question

    except Exception as e:
        # Log the error (implement proper logging)
        print(f"Error generating question: {str(e)}")
        # Return a fallback question if AI generation fails
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


@router.get("/current/{assessment_id}", response_model=QuestionResponse)
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
        question_text=await get_question(
            assessment.current_category, assessment_id, db
        ),  # Pass all required params
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

    return QuestionResponse(
        category=assessment.current_category,
        question_text=await get_question(
            assessment.current_category, answer.assessment_id, db
        ),  # Pass all required params
        assessment_id=assessment.id,
    )

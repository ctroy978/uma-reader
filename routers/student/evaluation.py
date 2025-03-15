# routers/evaluation.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import logging
import os

from database.session import get_db
from database.models import ActiveAssessment, User, Text, Chunk
from auth.middleware import require_user
from .questions import get_question
from .progression import update_category_on_result

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["student-evaluation"])

# Load environment variables and initialize AI model
load_dotenv()

ai_model = os.getenv("AI_MODEL", "gemini-2.0-flash")
model = GeminiModel(ai_model)

# model = GeminiModel("gemini-2.0-flash")


# Models
class AIEvaluation(BaseModel):
    """Model for AI evaluation response"""

    is_correct: bool = Field(..., description="Whether the answer is correct")
    feedback: str = Field(..., description="Feedback explanation for the student")


class AnswerSubmission(BaseModel):
    """Model for student answer submission"""

    answer: str
    question: str


class QuestionResponse(BaseModel):
    """Model for question format"""

    category: str
    question_text: str
    assessment_id: str
    from_cache: bool = False


class EvaluationResponse(BaseModel):
    """Model for API evaluation response"""

    is_correct: bool
    feedback: str
    next_question: Optional[QuestionResponse] = None
    can_progress: bool = False


def build_evaluation_prompt(
    context: str, question: str, answer: str, category: str
) -> str:
    """Build the context-aware evaluation prompt"""
    category_criteria = {
        "literal_basic": """
            Evaluation criteria:
            - Answer correctly identifies explicitly stated information
            - Response uses evidence directly from the text
            - Basic comprehension is demonstrated
        """,
        "literal_detailed": """
            Evaluation criteria:
            - Answer connects multiple details from the text
            - Response shows thorough understanding of explicit information
            - Supporting details are accurately referenced
        """,
        "inferential_simple": """
            Evaluation criteria:
            - Answer shows basic inferential thinking
            - Response connects ideas logically
            - Simple conclusions are supported by text evidence
        """,
        "inferential_complex": """
            Evaluation criteria:
            - Answer demonstrates deep analysis
            - Response shows advanced inferential thinking
            - Complex relationships between ideas are understood
            - Conclusions are well-supported with text evidence
        """,
    }

    return f"""
    You are an experienced reading teacher evaluating a student's answer.
    
    Text passage: "{context}"
    Question: "{question}"
    Student's answer: "{answer}"
    
    {category_criteria.get(category, "Evaluate the answer based on text evidence and comprehension.")}
    
    Provide:
    1. A boolean indicating if the answer is correct (is_correct)
    2. Constructive feedback explaining why (feedback)
    
    Keep feedback concise, specific, and encouraging.
    """


async def evaluate_answer_with_ai(
    context: str, question: str, answer: str, category: str
) -> AIEvaluation:
    """Generate AI evaluation using structured prompt"""
    try:
        agent = Agent(
            model=model,
            result_type=AIEvaluation,
            system_prompt="You are an experienced reading teacher providing evaluation feedback.",
        )

        prompt = build_evaluation_prompt(context, question, answer, category)
        result = await agent.run(prompt)

        if not result or not result.data:
            raise ValueError("Failed to generate evaluation")

        return result.data

    except Exception as e:
        logger.error(f"AI Evaluation Error: {str(e)}")
        # Fallback evaluation if AI fails
        return AIEvaluation(
            is_correct=False,
            feedback="Unable to evaluate answer. Please try again or contact support.",
        )


@router.post("/{assessment_id}", response_model=EvaluationResponse)
async def evaluate_answer(
    assessment_id: str,
    submission: AnswerSubmission,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Evaluate a student's answer and provide feedback."""
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

    try:
        # Get AI evaluation
        evaluation = await evaluate_answer_with_ai(
            current_chunk.content,
            submission.question,
            submission.answer,
            assessment.current_category,
        )

        # Store current category before update
        original_category = assessment.current_category

        # Update category based on correctness
        update_category_on_result(assessment, evaluation.is_correct)

        # Initialize next question as None
        next_question = None

        # If answer is wrong, always generate new question
        if not evaluation.is_correct:
            question_text, from_cache = await get_question(
                assessment.current_category, assessment_id, db
            )
            next_question = QuestionResponse(
                category=assessment.current_category,
                question_text=question_text,
                assessment_id=assessment_id,
                from_cache=from_cache,
            )

        # Can only progress to next chunk if answer is correct
        can_progress = evaluation.is_correct

        db.commit()

        return EvaluationResponse(
            is_correct=evaluation.is_correct,
            feedback=evaluation.feedback,
            next_question=next_question,
            can_progress=can_progress,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error evaluating answer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating answer: {str(e)}",
        )

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


class AnswerSubmission(BaseModel):
    """Model for student answer submission"""

    answer: str
    question: str
    is_pre_question: bool = False  # Flag to indicate if this is a pre-question


def build_evaluation_prompt(
    context: str,
    question: str,
    answer: str,
    category: str,
    grade_level: int = None,
    is_pre_question: bool = False,
) -> str:
    """Build a context-aware evaluation prompt for student answers

    Args:
        context: The text passage
        question: The question asked
        answer: The student's answer
        category: Question category for regular questions (e.g., literal_basic)
        grade_level: Optional grade level for age-appropriate feedback
        is_pre_question: Whether this is a pre-question or regular question

    Returns:
        A prompt for the AI model to evaluate the answer
    """
    # Define criteria based on whether this is a pre-question or regular question
    if is_pre_question:
        # Balanced pre-question criteria with emphasis on accuracy
        criteria = """
            EVALUATION CRITERIA FOR PRE-QUESTIONS:
            - Answer is CORRECT if it:
              1. Accurately captures an actual element, theme, or event from the specific passage
              2. Shows evidence the student has genuinely read THIS passage (not just making assumptions)
              3. Contains at least one accurate reference to the content (character, setting, action, or theme)
              4. Would only make sense as a response to THIS passage, not to any random text
              
            - Answer is INCORRECT if it:
              1. Misinterprets the fundamental content of the passage
              2. Mentions elements or events that aren't present in the passage
              3. Shows no evidence of having read this specific passage
              4. Is completely off-topic, irrelevant, or nonsensical
              5. Is so vague it could apply to almost any passage
              
            CRITICAL ACCURACY CHECK:
            Before marking an answer as correct, explicitly verify whether the elements mentioned 
            ACTUALLY APPEAR in the passage. If a student mentions "drowning" but the passage only 
            uses drowning as a metaphor and no one actually drowns, the answer is INCORRECT.
            
            The goal is to verify the student has actually read and understood what's genuinely 
            in the passage, not just made assumptions or guesses that sound plausible.
        """

        # Set system role for pre-questions
        system_role = "You are an experienced reading teacher evaluating a student's pre-question answer. You must CAREFULLY CHECK that their answer accurately reflects what's actually in the passage - not just what sounds plausible."
        question_type = "(This is a pre-question to check basic understanding)"

    else:
        # Regular question criteria based on category (unchanged)
        category_criteria = {
            "literal_basic": """
                Evaluation criteria:
                - Answer shows recognition of explicitly stated information from the text
                - Student demonstrates basic comprehension even if the answer is incomplete
                - Credit should be given for partial understanding
            """,
            "literal_detailed": """
                Evaluation criteria:
                - Answer identifies key details from the text, even if not all details are included
                - Student shows understanding of explicit information, even if explanation is imperfect
                - Accept answers that demonstrate the main point, even if supporting details are limited
            """,
            "inferential_simple": """
                Evaluation criteria:
                - Answer demonstrates basic inferential thinking, even if reasoning is not fully explained
                - Accept logical connections that show understanding, even if expressed simply
                - Credit reasonable interpretations even if different from the most obvious inference
            """,
            "inferential_complex": """
                Evaluation criteria:
                - Answer shows meaningful analysis, even if not exhaustive
                - Accept valid inferential thinking that demonstrates understanding of underlying concepts
                - Credit thoughtful interpretations that are supported by the text, even if imperfectly expressed
            """,
        }

        # Use category-specific criteria if available, or default criteria
        criteria = category_criteria.get(
            category, "Evaluate the answer based on text evidence and comprehension."
        )

        # Add standard evaluation guidelines for regular questions
        criteria += """
            IMPORTANT EVALUATION GUIDELINES:
            - Consider answers correct if they demonstrate understanding of the core concept, even if imperfectly expressed
            - Accept different wording, phrasing, or vocabulary that still shows comprehension
            - Give credit for partial understanding when the main point is captured
            - Be flexible with grammar, spelling, or phrasing issues if they don't interfere with meaning
            - Consider the answer correct if it contains the essential elements, even if not fully developed
            
            Evaluation threshold:
            - Correct (TRUE): The answer demonstrates understanding of the key concepts, even if imperfect
            - Incorrect (FALSE): The answer shows fundamental misunderstanding or contains no relevant information
        """

        # Set system role for regular questions
        system_role = (
            "You are an experienced reading teacher providing evaluation feedback."
        )
        question_type = ""

    # Add grade-level guidance if available
    grade_guidance = ""
    if grade_level:
        grade_guidance = f"\nThis student is at grade level {grade_level}. Evaluate appropriately for this age group."

    # Build the complete prompt
    return f"""
    {system_role}
    
    Text passage: "{context}"
    Question: "{question}" {question_type}
    Student's answer: "{answer}"
    
    {criteria}
    {grade_guidance}
    
    Provide:
    1. A boolean indicating if the answer is correct (is_correct) using the threshold described above
    2. {"Brief, encouraging feedback (2-3 sentences)" if is_pre_question else "Constructive, encouraging feedback"} explaining your evaluation
    
    Keep feedback concise, specific, and positive. Acknowledge what the student did well even when giving critical feedback.
    
    {'For pre-questions, carefully verify that the student\'s answer mentions elements ACTUALLY PRESENT in the passage, not just plausible guesses.' if is_pre_question else ''}
    """


async def evaluate_answer_with_ai(
    context: str,
    question: str,
    answer: str,
    category: str,
    grade_level: int = None,
    is_pre_question: bool = False,
) -> AIEvaluation:
    """Generate AI evaluation using structured prompt"""
    try:
        agent = Agent(
            model=model,
            result_type=AIEvaluation,
            system_prompt="You are an experienced reading teacher providing evaluation feedback.",
        )

        prompt = build_evaluation_prompt(
            context, question, answer, category, grade_level, is_pre_question
        )
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
    """Evaluate a student's answer and provide feedback.

    Handles both regular questions and pre-questions (basic comprehension checks).
    """
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

    # Get text to determine grade level (for evaluation context)
    text = db.query(Text).get(assessment.text_id)
    grade_level = text.grade_level if text else None

    try:
        # Determine if this is a pre-question or regular question
        is_pre_question = (
            submission.is_pre_question
            if hasattr(submission, "is_pre_question")
            else False
        )

        # Get AI evaluation with appropriate criteria
        evaluation = await evaluate_answer_with_ai(
            context=current_chunk.content,
            question=submission.question,
            answer=submission.answer,
            category=assessment.current_category,
            grade_level=grade_level,
            is_pre_question=is_pre_question,
        )

        # Initialize response parameters
        next_question = None
        can_progress = False

        # Handle pre-questions differently from regular questions
        if is_pre_question:
            # For pre-questions:
            # - If correct, allow progress to regular question
            # - If incorrect, no progression, no category change
            can_progress = evaluation.is_correct

            # If pre-question is correct, fetch the regular question
            if evaluation.is_correct:
                question_text, from_cache = await get_question(
                    assessment.current_category, assessment_id, db
                )
                next_question = QuestionResponse(
                    category=assessment.current_category,
                    question_text=question_text,
                    assessment_id=assessment_id,
                    from_cache=from_cache,
                )
        else:
            # For regular questions:
            # - Update category progression based on correctness
            # - Only allow chunk progression if correct
            original_category = assessment.current_category

            # Update category based on correctness
            update_category_on_result(assessment, evaluation.is_correct)

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

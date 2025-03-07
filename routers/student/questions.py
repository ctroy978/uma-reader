# routers/questions.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, Tuple
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import uuid
import logging

from database.session import get_db
from database.models import ActiveAssessment, User, Text, Chunk, QuestionCache
from auth.middleware import require_user

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Load environment variables and initialize AI model
load_dotenv()
model = GeminiModel("gemini-2.0-flash")


class Question(BaseModel):
    """Model for AI-generated questions"""

    question: str = Field(..., description="The generated question text")
    category: str = Field(..., description="The question category")
    grade_level: int = Field(..., description="Target grade level")


class QuestionResponse(BaseModel):
    """API response model"""

    category: str
    question_text: str
    assessment_id: str
    from_cache: bool = False

    class Config:
        from_attributes = True


def get_category_prompt(category: str) -> str:
    """Get specific prompt instructions based on question category"""
    prompts = {
        "literal_basic": """
            Create a basic comprehension question that:
            - Tests understanding of explicitly stated information
            - Focuses on main ideas or key details directly from the text
            - Can be answered using specific words/phrases from the passage
            - Is appropriate for the given grade level
        """,
        "literal_detailed": """
            Create a detailed comprehension question that:
            - Tests understanding of specific details and supporting information
            - Requires connecting multiple explicit details from the text
            - Focuses on how ideas are related or organized
            - Matches the complexity appropriate for the grade level
        """,
        "inferential_simple": """
            Create a simple inferential question that:
            - Requires basic reasoning beyond the explicit text
            - Asks students to make straightforward connections
            - Tests understanding of cause and effect or basic relationships
            - Is suitable for building initial inferential skills
        """,
        "inferential_complex": """
            Create a complex inferential question that:
            - Requires deeper analysis and interpretation
            - Tests understanding of themes, motives, or abstract concepts
            - Asks students to synthesize information across the text
            - Challenges students to think critically at their grade level
        """,
    }
    return prompts.get(
        category,
        "Create a reading comprehension question appropriate for the text and grade level.",
    )


async def generate_ai_question(
    category: str, grade_level: int, content: str
) -> Question:
    """Generate a question using the AI model"""
    try:
        agent = Agent(
            model=model,
            result_type=Question,
            system_prompt="You are an AI reading comprehension question generator.",
        )

        prompt = f"""
        You are an experienced reading teacher creating a {category} question for grade {grade_level}.
        
        Text content: "{content}"
        
        {get_category_prompt(category)}
        
        Important guidelines:
        - Question should be clear and unambiguous
        - Avoid yes/no questions
        - Use grade-appropriate vocabulary
        - Focus on understanding rather than memorization
        - Questions should be answerable using the provided text
        
        Generate a single question that meets these criteria.
        """

        result = await agent.run(prompt)

        if not result or not result.data:
            raise ValueError("Failed to generate question")

        return result.data

    except Exception as e:
        logger.error(f"AI Question Generation Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate question",
        )


# This is the function that evaluation.py will import
async def get_question(
    category: str, assessment_id: str, db: Session, use_cache: bool = True
) -> Tuple[str, bool]:
    """
    Get question text for a given category and assessment.

    Args:
        category: Question category (e.g., literal_basic)
        assessment_id: ID of the active assessment
        db: Database session
        use_cache: Whether to use/update cache (default: True)

    Returns:
        Tuple of (question_text, from_cache)
    """
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

    current_chunk = db.query(Chunk).get(assessment.current_chunk_id)
    text = db.query(Text).get(assessment.text_id)

    if not current_chunk or not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current chunk or text not found",
        )

    # Check cache first if using cache
    if use_cache:
        cached_question = (
            db.query(QuestionCache)
            .filter(
                QuestionCache.chunk_id == current_chunk.id,
                QuestionCache.question_category == category,
                QuestionCache.grade_level == text.grade_level,
                QuestionCache.is_deleted == False,
            )
            .first()
        )

        if cached_question:
            # Update access metrics
            cached_question.increment_access()
            db.commit()
            logger.info(
                f"Cache hit: Using cached question for chunk {current_chunk.id}, category {category}"
            )
            return cached_question.question_text, True

    try:
        # Cache miss - generate new question
        question = await generate_ai_question(
            category=category,
            grade_level=text.grade_level,
            content=current_chunk.content,
        )

        if use_cache:
            # Store in cache
            new_cached_question = QuestionCache(
                id=str(uuid.uuid4()),
                chunk_id=current_chunk.id,
                question_category=category,
                grade_level=text.grade_level,
                question_text=question.question,
                access_count=1,
                last_accessed=datetime.now(timezone.utc),
            )

            db.add(new_cached_question)
            db.commit()
            logger.info(
                f"Cache miss: Generated new question for chunk {current_chunk.id}, category {category}"
            )

        return question.question, False
    except Exception as e:
        db.rollback()
        logger.error(f"Error generating/caching question: {str(e)}")
        # Fallback question if AI generation fails
        return (
            f"Based on the text, explain the main concepts presented in this section.",
            False,
        )


@router.get("/current/{assessment_id}", response_model=QuestionResponse)
async def get_current_question(
    assessment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
    use_cache: bool = True,
):
    """Get the current question for an assessment with optional caching"""
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

    question_text, from_cache = await get_question(
        assessment.current_category, assessment_id, db, use_cache
    )

    return QuestionResponse(
        category=assessment.current_category,
        question_text=question_text,
        assessment_id=assessment_id,
        from_cache=from_cache,
    )

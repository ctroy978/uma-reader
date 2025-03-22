# routers/questions.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import uuid
import logging
import os
from enum import Enum
from typing import Optional, Tuple, Dict, List

from database.session import get_db
from database.models import ActiveAssessment, User, Text, Chunk, QuestionCache
from auth.middleware import require_user

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Load environment variables and initialize AI model
load_dotenv()

ai_model = os.getenv("AI_MODEL", "gemini-2.0-flash")
model = GeminiModel(ai_model)


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


def get_category_prompt(category: str, grade_level: int) -> str:
    """Get specific prompt instructions based on question category and grade level"""

    # Define grade band appropriate instructions
    grade_guidance = ""
    if 2 <= grade_level <= 5:  # Elementary
        grade_guidance = """
            - Use simple, direct language with 1-2 short sentences
            - Vocabulary should be concrete and familiar to elementary students
            - Questions should be answerable in 1-3 sentences
            - Aim for a reading level 1-2 grades below the target grade
        """
    elif 6 <= grade_level <= 8:  # Middle School
        grade_guidance = """
            - Use clear language with 1-2 concise sentences
            - Avoid complex clauses and academic jargon
            - Questions should be answerable in 2-4 sentences
            - Aim for a reading level at or slightly below the target grade
        """
    else:  # High School (9-12)
        grade_guidance = """
            - Use straightforward language with no more than 2 sentences
            - Avoid unnecessarily sophisticated vocabulary
            - Questions should be focused and specific
            - Aim for a reading level appropriate to the target grade
        """

    # Define category-specific prompts
    prompts = {
        "literal_basic": f"""
            Create a basic comprehension question that:
            - Asks directly about key information explicitly stated in the text
            - Uses a single, clear sentence
            - Can be answered by pointing to specific words or phrases in the text
            
            {grade_guidance}
            
            Example good questions:
            - "What did [character] do when [event happened]?"
            - "What is the main setting of this passage?"
            - "According to the text, what caused [event]?"
        """,
        "literal_detailed": f"""
            Create a detailed comprehension question that:
            - Asks about specific details from the text
            - Uses a single, direct sentence
            - Focuses on important supporting information
            
            {grade_guidance}
            
            Example good questions:
            - "What evidence does the text provide to support [main idea]?"
            - "What details does the author use to describe [element]?"
            - "How does the passage explain the process of [topic]?"
        """,
        "inferential_simple": f"""
            Create a simple inferential question that:
            - Asks about a straightforward conclusion not directly stated
            - Uses a clear, focused sentence
            - Can be answered by connecting information from the text
            
            {grade_guidance}
            
            Example good questions:
            - "How might [character] feel about [event] based on the text?"
            - "What is likely to happen next based on this passage?"
            - "Why did [character] probably decide to [action]?"
        """,
        "inferential_complex": f"""
            Create a thought-provoking but clear inferential question that:
            - Asks about deeper meaning or connections
            - Uses simple language despite testing complex thinking
            - Remains focused on a specific aspect of the text
            
            {grade_guidance}
            
            Example good questions:
            - "What lesson might the author want readers to learn from this passage?"
            - "How does [character's] response to [situation] reveal their personality?"
            - "What does [symbol/element] represent in this text?"
        """,
    }

    return prompts.get(
        category,
        f"Create a clear, focused reading question appropriate for grade {grade_level}. {grade_guidance}",
    )


async def generate_ai_question(
    category: str, grade_level: int, content: str
) -> Question:
    """Generate a question using the AI model with improved clarity and grade-level appropriateness"""
    try:
        agent = Agent(
            model=model,
            result_type=Question,
            system_prompt="You are an AI reading teacher creating clear, grade-appropriate questions.",
        )

        prompt = f"""
        You are creating a reading comprehension question for grade {grade_level} students.
        
        Text content: "{content}"
        
        {get_category_prompt(category, grade_level)}
        
        IMPORTANT GUIDELINES:
        - Keep the question SHORT (no more than 15-20 words)
        - Use ONE sentence only
        - Be direct and specific - avoid unnecessary words
        - Make sure the question has a clear focus
        - Use simple language that grade {grade_level} students will understand
        - Avoid complex sentence structures with multiple clauses
        - Questions must be answerable using the provided text
        - Avoid yes/no questions
        
        Before submitting your response, verify:
        1. Is this question clear and focused?
        2. Is it appropriate for grade {grade_level}?
        3. Is it expressed in a single, concise sentence?
        4. Would a grade {grade_level} student understand what is being asked?
        5. Can this question be answered ONLY using the specific text provided?
        6. Have you checked that any quantities mentioned are accurate (e.g., "three animals" only if exactly three are mentioned)?


        
        Generate a single, short question that meets these criteria.
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
            f"Based on the text, what is the main idea of this section?",
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


# Add these imports if they aren't already present
from enum import Enum
from typing import Optional, Tuple, Dict, List

## Add these imports if they aren't already present
from enum import Enum
from typing import Optional, Tuple, Dict, List


# Define pre-question types based on text type
class PreQuestionType(str, Enum):
    NARRATIVE = "NARRATIVE"
    INFORMATIONAL = "INFORMATIONAL"
    OTHER = "OTHER"


# Pre-defined questions by text type
PRE_QUESTIONS: Dict[str, List[str]] = {
    "NARRATIVE": [
        "Summarize what happens in this passage.",
        "Briefly describe the key events in this section.",
    ],
    "INFORMATIONAL": [
        "What is being explained in this section?",
        "Summarize the key points presented in this passage.",
        "What information does this section provide?",
    ],
    "OTHER": [
        "Summarize the main ideas in this passage.",
        "What is this section mainly about?",
        "What are the key points from this section?",
    ],
}


class PreQuestionResponse(BaseModel):
    """Response model for pre-questions"""

    question_text: str
    assessment_id: str
    is_pre_question: bool = True

    class Config:
        from_attributes = True


async def get_pre_question(assessment_id: str, db: Session) -> str:
    """
    Get a pre-question for the given assessment

    Args:
        assessment_id: ID of the active assessment
        db: Database session

    Returns:
        A pre-question text appropriate for the text type
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

    text = db.query(Text).get(assessment.text_id)

    if not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Text not found"
        )

    # Choose appropriate pre-question based on text type
    text_type = text.type_name
    # Map persuasive to informational for pre-questions
    if text_type == "PERSUASIVE":
        text_type = "INFORMATIONAL"
    if text_type not in PRE_QUESTIONS:
        text_type = "OTHER"

    # Select a random question from the appropriate list
    import random

    question = random.choice(PRE_QUESTIONS[text_type])

    logger.info(f"Generated pre-question for text type {text_type}: {question}")
    return question


@router.get("/pre-question/{assessment_id}", response_model=PreQuestionResponse)
async def get_pre_question_endpoint(
    assessment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get a pre-question for the current chunk to verify reading comprehension"""
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

    pre_question_text = await get_pre_question(assessment_id, db)

    return PreQuestionResponse(
        question_text=pre_question_text,
        assessment_id=assessment_id,
    )

# routers/student/simplify.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

from database.session import get_db
from database.models.text import Text, Chunk
from database.models.user import User
from database.models.assessment import ActiveAssessment
from database.models.simplified_text import SimplifiedChunk
from auth.middleware import require_user

router = APIRouter(tags=["text-simplification"])

# Load environment variables and initialize AI model
load_dotenv()

ai_model = os.getenv("AI_MODEL", "gemini-2.0-flash-lite")
model = GeminiModel(ai_model)

# model = GeminiModel("gemini-2.0-flash")


class SimplifiedText(BaseModel):
    """Model for AI-generated simplified text"""

    simplified_text: str = Field(..., description="The simplified version of the text")
    target_grade_level: int = Field(
        ..., description="The target grade level for simplification"
    )


class SimplifyRequest(BaseModel):
    """Request model for text simplification"""

    text_content: str
    current_grade_level: int


class SimplifyResponse(BaseModel):
    """API response model"""

    simplified_text: str
    original_grade_level: int
    target_grade_level: int
    is_cached: bool = False

    class Config:
        from_attributes = True


async def generate_simplified_text(content: str, current_grade: int) -> SimplifiedText:
    """Generate simplified text using the AI model"""
    try:
        # Calculate target grade level (3 levels below or minimum grade 1)
        target_grade = max(1, current_grade - 3)

        agent = Agent(
            model=model,
            result_type=SimplifiedText,
            system_prompt="You are an AI text simplification assistant for students.",
        )

        prompt = f"""
        Simplify the following text from grade {current_grade} level to grade {target_grade} level.
        Make it easier to understand while preserving the key information and meaning.
        
        Original text: "{content}"
        
        Guidelines:
        - Use simpler vocabulary appropriate for grade {target_grade}
        - Use shorter sentences
        - Explain difficult concepts in simple terms
        - Maintain the same overall meaning and key information
        - Keep a similar structure (paragraphs, dialogue, etc.)
        
        Return only the simplified text without any additional explanation.
        """

        result = await agent.run(prompt)

        if not result or not result.data:
            raise ValueError("Failed to generate simplified text")

        return result.data

    except Exception as e:
        print(f"AI Text Simplification Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to simplify text",
        )


@router.post("/text", response_model=SimplifyResponse)
async def simplify_text(
    request: SimplifyRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Simplify text to a lower grade level"""
    try:
        simplified = await generate_simplified_text(
            content=request.text_content, current_grade=request.current_grade_level
        )

        return SimplifyResponse(
            simplified_text=simplified.simplified_text,
            original_grade_level=request.current_grade_level,
            target_grade_level=simplified.target_grade_level,
            is_cached=False,
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error simplifying text: {str(e)}",
        )


@router.get("/chunk/{assessment_id}", response_model=SimplifyResponse)
async def simplify_current_chunk(
    assessment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Simplify the current chunk for an active assessment"""
    # Find the active assessment
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

    # Get the current chunk and text
    current_chunk = db.query(Chunk).get(assessment.current_chunk_id)
    text = db.query(Text).get(assessment.text_id)

    if not current_chunk or not text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current chunk or text not found",
        )

    # Calculate target grade level (3 levels below or minimum grade 1)
    target_grade = max(1, text.grade_level - 3)

    # Check if we already have a cached simplified version
    cached_version = (
        db.query(SimplifiedChunk)
        .filter(
            SimplifiedChunk.chunk_id == current_chunk.id,
            SimplifiedChunk.target_grade_level == target_grade,
            SimplifiedChunk.is_deleted == False,
        )
        .first()
    )

    # If we have a cached version, return it and increment the access count
    if cached_version:
        print(f"Using cached simplified text for chunk {current_chunk.id}")
        cached_version.increment_access_count()
        db.commit()

        return SimplifyResponse(
            simplified_text=cached_version.simplified_content,
            original_grade_level=cached_version.original_grade_level,
            target_grade_level=cached_version.target_grade_level,
            is_cached=True,
        )

    # Otherwise, generate a new simplified version
    try:
        simplified = await generate_simplified_text(
            content=current_chunk.content, current_grade=text.grade_level
        )

        # Create a new SimplifiedChunk record
        new_cached_version = SimplifiedChunk(
            chunk_id=current_chunk.id,
            original_grade_level=text.grade_level,
            target_grade_level=simplified.target_grade_level,
            simplified_content=simplified.simplified_text,
            access_count=1,  # Initial access
        )

        db.add(new_cached_version)
        db.commit()

        print(f"Created new cached simplified text for chunk {current_chunk.id}")

        return SimplifyResponse(
            simplified_text=simplified.simplified_text,
            original_grade_level=text.grade_level,
            target_grade_level=simplified.target_grade_level,
            is_cached=False,
        )

    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error simplifying text: {str(e)}",
        )

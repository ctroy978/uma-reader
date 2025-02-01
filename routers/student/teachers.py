from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from database.session import get_db
from database.models import User, Text, Role
from database.models.references import TextForm, Genre, PrimaryType

router = APIRouter(tags=["Student Teacher Access"])


# Pydantic models for responses
class TeacherResponse(BaseModel):
    id: str
    full_name: str
    text_count: int

    class Config:
        from_attributes = True


class TeacherStatsResponse(BaseModel):
    total_texts: int
    grade_levels: List[int]
    forms: List[str]
    genres: List[str]

    class Config:
        from_attributes = True


# Get list of active teachers with text counts
@router.get("/", response_model=List[TeacherResponse])
async def list_teachers(db: Session = Depends(get_db)):
    """
    Get list of all teachers who have published texts
    """
    # Query teachers with text counts
    teachers = (
        db.query(User, func.count(Text.id).label("text_count"))
        .join(Role, User.role_name == Role.role_name)
        .outerjoin(Text, (Text.teacher_id == User.id) & (Text.is_deleted == False))
        .filter(User.role_name == "TEACHER", User.is_deleted == False)
        .group_by(User.id)
        .having(func.count(Text.id) > 0)  # Only include teachers with texts
        .all()
    )

    return [
        TeacherResponse(
            id=teacher.User.id,
            full_name=teacher.User.full_name,
            text_count=teacher.text_count,
        )
        for teacher in teachers
    ]


# Get teacher stats
@router.get("/{teacher_id}/stats", response_model=TeacherStatsResponse)
async def get_teacher_stats(teacher_id: str, db: Session = Depends(get_db)):
    """
    Get statistics about a teacher's texts
    """
    # Verify teacher exists and has texts
    teacher = (
        db.query(User)
        .filter(
            User.id == teacher_id, User.role_name == "TEACHER", User.is_deleted == False
        )
        .first()
    )

    if not teacher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Teacher not found"
        )

    # Get text statistics
    texts = (
        db.query(Text)
        .filter(Text.teacher_id == teacher_id, Text.is_deleted == False)
        .all()
    )

    if not texts:
        return TeacherStatsResponse(total_texts=0, grade_levels=[], forms=[], genres=[])

    # Collect unique values
    grade_levels = sorted(list(set(text.grade_level for text in texts)))
    forms = sorted(list(set(text.form_name for text in texts)))

    # Collect all genres used by teacher
    genres = (
        db.query(Genre.genre_name)
        .join(Text.genres)
        .filter(Text.teacher_id == teacher_id, Text.is_deleted == False)
        .distinct()
        .all()
    )
    genre_names = sorted([genre[0] for genre in genres])

    return TeacherStatsResponse(
        total_texts=len(texts),
        grade_levels=grade_levels,
        forms=forms,
        genres=genre_names,
    )


# Pydantic model for text list response
class TextListResponse(BaseModel):
    id: str
    title: str
    grade_level: int
    form: str
    primary_type: str
    genres: List[str]

    class Config:
        from_attributes = True


class TextListPaginatedResponse(BaseModel):
    texts: List[TextListResponse]
    total: int
    page: int

    class Config:
        from_attributes = True


# Get teacher's texts with filtering
@router.get("/{teacher_id}/texts", response_model=TextListPaginatedResponse)
async def list_teacher_texts(
    teacher_id: str,
    grade_level: Optional[int] = Query(None, ge=2, le=12),
    form: Optional[str] = Query(None),
    primary_type: Optional[str] = Query(None),
    genres: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Get filtered list of texts from a specific teacher
    """
    # Verify teacher exists
    teacher = (
        db.query(User)
        .filter(
            User.id == teacher_id, User.role_name == "TEACHER", User.is_deleted == False
        )
        .first()
    )

    if not teacher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Teacher not found"
        )

    # Build base query
    query = db.query(Text).filter(
        Text.teacher_id == teacher_id, Text.is_deleted == False
    )

    # Apply filters
    if grade_level is not None:
        query = query.filter(Text.grade_level == grade_level)

    if form is not None:
        query = query.filter(Text.form_name == form)

    if primary_type is not None:
        query = query.filter(Text.type_name == primary_type)

    if genres:
        for genre in genres:
            query = query.filter(Text.genres.any(Genre.genre_name == genre))

    # Get total count before pagination
    total = query.count()

    # Apply pagination
    offset = (page - 1) * per_page
    texts = query.order_by(Text.created_at.desc()).offset(offset).limit(per_page).all()

    return TextListPaginatedResponse(
        texts=[
            TextListResponse(
                id=text.id,
                title=text.title,
                grade_level=text.grade_level,
                form=text.form_name,
                primary_type=text.type_name,
                genres=[genre.genre_name for genre in text.genres],
            )
            for text in texts
        ],
        total=total,
        page=page,
    )

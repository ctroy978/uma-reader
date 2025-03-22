# app/routers/teacher/bypass.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator
from typing import Optional
import re

from database.session import get_db
from database.models import User, TeacherBypassCode
from auth.middleware import require_user

router = APIRouter(tags=["Teacher Bypass"])


class BypassCodeSchema(BaseModel):
    """Schema for bypass code management"""

    bypass_code: str

    @field_validator("bypass_code")
    @classmethod
    def validate_bypass_code(cls, v):
        if not re.match(r"^\d{4}$", v):
            raise ValueError("Bypass code must be exactly 4 digits")
        return v

    class Config:
        from_attributes = True


class BypassCodeResponse(BaseModel):
    """Response schema for bypass code operations"""

    message: str
    bypass_code: Optional[str] = None

    class Config:
        from_attributes = True


@router.post("/bypass-code", response_model=BypassCodeResponse)
async def set_bypass_code(
    bypass_code_data: BypassCodeSchema,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Set or update a teacher's bypass code"""
    # Verify user is a teacher
    if user.role_name != "TEACHER" and user.role_name != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers can set bypass codes",
        )

    # Check for existing bypass code
    existing_code = (
        db.query(TeacherBypassCode)
        .filter(TeacherBypassCode.teacher_id == user.id)
        .first()
    )

    try:
        if existing_code:
            # Update existing code
            existing_code.bypass_code = bypass_code_data.bypass_code
            existing_code.is_active = True
            db.commit()
            return BypassCodeResponse(
                message="Bypass code updated successfully",
                bypass_code=existing_code.bypass_code,
            )
        else:
            # Create new bypass code
            new_bypass_code = TeacherBypassCode(
                teacher_id=user.id,
                bypass_code=bypass_code_data.bypass_code,
                is_active=True,
            )
            db.add(new_bypass_code)
            db.commit()
            return BypassCodeResponse(
                message="Bypass code created successfully",
                bypass_code=new_bypass_code.bypass_code,
            )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting bypass code: {str(e)}",
        )


@router.get("/bypass-code", response_model=BypassCodeResponse)
async def get_bypass_code(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get a teacher's current bypass code"""
    # Verify user is a teacher
    if user.role_name != "TEACHER" and user.role_name != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers can access bypass codes",
        )

    # Get existing bypass code
    bypass_code = (
        db.query(TeacherBypassCode)
        .filter(
            TeacherBypassCode.teacher_id == user.id, TeacherBypassCode.is_active == True
        )
        .first()
    )

    if not bypass_code:
        return BypassCodeResponse(
            message="No bypass code set. Please create one.",
            bypass_code=None,
        )

    return BypassCodeResponse(
        message="Bypass code retrieved successfully",
        bypass_code=bypass_code.bypass_code,
    )


@router.delete("/bypass-code", response_model=BypassCodeResponse)
async def deactivate_bypass_code(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Deactivate a teacher's bypass code"""
    # Verify user is a teacher
    if user.role_name != "TEACHER" and user.role_name != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers can manage bypass codes",
        )

    # Get existing bypass code
    bypass_code = (
        db.query(TeacherBypassCode)
        .filter(TeacherBypassCode.teacher_id == user.id)
        .first()
    )

    if not bypass_code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No bypass code found to deactivate",
        )

    try:
        bypass_code.is_active = False
        db.commit()
        return BypassCodeResponse(
            message="Bypass code deactivated successfully",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deactivating bypass code: {str(e)}",
        )

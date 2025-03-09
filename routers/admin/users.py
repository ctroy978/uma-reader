from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Annotated, List
from pydantic import BaseModel, UUID4
from sqlalchemy import or_

from database.session import get_db
from database.models import User
from auth.middleware import require_admin

router = APIRouter(prefix="/users", tags=["Admin"])


# Response Models
from datetime import datetime


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    role_name: str
    is_deleted: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        # Configure JSON serialization
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class UserRoleUpdate(BaseModel):
    role_name: str


# Routes
@router.get("", response_model=List[UserResponse])
async def get_users(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
    role: str = None,
    status: str = None,
    search: str = None,
):
    """Get users with optional filters"""
    query = db.query(User)

    if role:
        query = query.filter(User.role_name == role)
    if status:
        if status == "active":
            query = query.filter(User.is_deleted == False)
        elif status == "deleted":
            query = query.filter(User.is_deleted == True)
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                User.username.ilike(search_term),
                User.email.ilike(search_term),
                User.full_name.ilike(search_term),
            )
        )

    users = query.all()
    return users


@router.put("/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: str,
    role_update: UserRoleUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
):
    """Update a user's role"""
    # Prevent admin from changing their own role
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Validate role
    if role_update.role_name not in ["ADMIN", "TEACHER", "STUDENT"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role"
        )

    user.role_name = role_update.role_name
    db.commit()
    db.refresh(user)
    return user


@router.put("/{user_id}/toggle-delete", response_model=UserResponse)
async def toggle_user_delete(
    user_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
):
    """Toggle user's deleted status"""
    # Prevent admin from deleting themselves
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    user.is_deleted = not user.is_deleted
    db.commit()
    db.refresh(user)
    return user

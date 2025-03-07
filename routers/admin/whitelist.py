# app/routers/admin/whitelist.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional

from database.session import get_db
from database.models import EmailWhitelist, WhitelistType
from auth.middleware import require_admin
from services.whitelist_service import WhitelistService

router = APIRouter(tags=["Admin - Whitelist Management"])


class WhitelistEntryCreate(BaseModel):
    value: str = Field(..., min_length=3, max_length=255)
    type: WhitelistType
    description: Optional[str] = None

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "value": "csd8.info",
                    "type": "domain",
                    "description": "School district domain",
                },
                {
                    "value": "special.student@example.com",
                    "type": "email",
                    "description": "Special case student",
                },
            ]
        }


class WhitelistEntryResponse(BaseModel):
    id: str
    value: str
    type: WhitelistType
    description: Optional[str] = None

    class Config:
        from_attributes = True


@router.get("/whitelist", response_model=List[WhitelistEntryResponse])
async def get_whitelist(
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),  # Ensure only admins can access
):
    """Get all whitelist entries"""
    return WhitelistService.get_all_whitelist_entries(db)


@router.post(
    "/whitelist",
    response_model=WhitelistEntryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_whitelist_entry(
    entry: WhitelistEntryCreate,
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),  # Ensure only admins can access
):
    """Add a new whitelist entry"""
    try:
        return WhitelistService.add_to_whitelist(
            entry.value, entry.type, entry.description, db
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/whitelist/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_whitelist_entry(
    entry_id: str,
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),  # Ensure only admins can access
):
    """Remove a whitelist entry"""
    result = WhitelistService.remove_from_whitelist(entry_id, db)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Whitelist entry with id {entry_id} not found",
        )

    return None  # 204 No Content

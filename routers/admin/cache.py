# routers/admin/cache.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime, timedelta

from database.session import get_db
from database.models.simplified_text import SimplifiedChunk
from database.models.user import User
from auth.middleware import (
    require_admin,
)  # This will ensure only admins can access these endpoints

router = APIRouter(tags=["cache-management"])


class SimplifiedChunkInfo(BaseModel):
    """Information about a cached simplified chunk"""

    id: str
    chunk_id: str
    original_grade_level: int
    target_grade_level: int
    access_count: int
    last_accessed: datetime
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/simplified-text", response_model=List[SimplifiedChunkInfo])
async def list_simplified_text_cache(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    min_access: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),  # Ensure admin access
):
    """List cached simplified text chunks"""
    query = db.query(SimplifiedChunk).filter(SimplifiedChunk.is_deleted == False)

    if min_access > 0:
        query = query.filter(SimplifiedChunk.access_count >= min_access)

    cache_entries = (
        query.order_by(SimplifiedChunk.access_count.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Convert to response model format
    result = []
    for entry in cache_entries:
        result.append(
            SimplifiedChunkInfo(
                id=entry.id,
                chunk_id=entry.chunk_id,
                original_grade_level=entry.original_grade_level,
                target_grade_level=entry.target_grade_level,
                access_count=entry.access_count,
                last_accessed=entry.updated_at,
                created_at=entry.created_at,
            )
        )

    return result


@router.delete("/simplified-text/{cache_id}")
async def delete_simplified_text_cache(
    cache_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),  # Ensure admin access
):
    """Delete a specific simplified text cache entry"""
    cache_entry = db.query(SimplifiedChunk).get(cache_id)

    if not cache_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cache entry not found",
        )

    # Soft delete the entry
    cache_entry.is_deleted = True
    db.commit()

    return {"message": f"Cache entry {cache_id} has been deleted"}


@router.delete("/simplified-text/cleanup/unused")
async def cleanup_unused_simplified_text_cache(
    min_days_old: int = Query(30, ge=1),
    max_access_count: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),  # Ensure admin access
):
    """Clean up unused simplified text cache entries

    - Deletes entries older than min_days_old
    - With access_count less than or equal to max_access_count
    """
    cutoff_date = datetime.utcnow() - timedelta(days=min_days_old)

    # Find entries to delete
    query = db.query(SimplifiedChunk).filter(
        SimplifiedChunk.is_deleted == False,
        SimplifiedChunk.created_at < cutoff_date,
        SimplifiedChunk.access_count <= max_access_count,
    )

    # Get count for response
    count = query.count()

    # Soft delete the entries
    query.update({"is_deleted": True}, synchronize_session=False)
    db.commit()

    return {
        "message": f"Deleted {count} unused cache entries",
        "deleted_count": count,
    }

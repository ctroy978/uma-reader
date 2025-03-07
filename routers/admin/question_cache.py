# routers/admin/question_cache.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime, timedelta

from database.session import get_db
from database.models import QuestionCache
from database.models.user import User
from auth.middleware import require_admin

router = APIRouter(tags=["question-cache-management"])


class QuestionCacheInfo(BaseModel):
    """Information about a cached question"""

    id: str
    chunk_id: str
    question_category: str
    grade_level: int
    access_count: int
    last_accessed: datetime
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/questions", response_model=List[QuestionCacheInfo])
async def list_question_cache(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    min_access: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),  # Ensure admin access
):
    """List cached questions"""
    query = db.query(QuestionCache).filter(QuestionCache.is_deleted == False)

    if min_access > 0:
        query = query.filter(QuestionCache.access_count >= min_access)

    cache_entries = (
        query.order_by(QuestionCache.access_count.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Convert to response model format
    return cache_entries


@router.delete("/questions/{cache_id}")
async def delete_question_cache(
    cache_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),  # Ensure admin access
):
    """Delete a specific question cache entry"""
    cache_entry = db.query(QuestionCache).get(cache_id)

    if not cache_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cache entry not found",
        )

    # Soft delete the entry
    cache_entry.is_deleted = True
    db.commit()

    return {"message": f"Cache entry {cache_id} has been deleted"}


@router.delete("/questions/cleanup/unused")
async def cleanup_unused_question_cache(
    min_days_old: int = Query(30, ge=1),
    max_access_count: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),  # Ensure admin access
):
    """Clean up unused question cache entries

    - Deletes entries older than min_days_old
    - With access_count less than or equal to max_access_count
    """
    cutoff_date = datetime.utcnow() - timedelta(days=min_days_old)

    # Find entries to delete
    query = db.query(QuestionCache).filter(
        QuestionCache.is_deleted == False,
        QuestionCache.created_at < cutoff_date,
        QuestionCache.access_count <= max_access_count,
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


@router.get("/statistics")
async def get_cache_statistics(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),  # Ensure admin access
):
    """Get statistics about the question cache"""
    total_entries = (
        db.query(QuestionCache).filter(QuestionCache.is_deleted == False).count()
    )

    # Get total hits (sum of access_count)
    result = (
        db.query(db.func.sum(QuestionCache.access_count).label("total_hits"))
        .filter(QuestionCache.is_deleted == False)
        .first()
    )

    total_hits = result.total_hits or 0

    # Get most used entries
    most_used = (
        db.query(QuestionCache)
        .filter(QuestionCache.is_deleted == False)
        .order_by(QuestionCache.access_count.desc())
        .limit(5)
        .all()
    )

    # Format most used entries
    most_used_list = [
        {
            "id": entry.id,
            "category": entry.question_category,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed,
        }
        for entry in most_used
    ]

    # Calculate hit rate
    hit_rate = (total_hits / total_entries) if total_entries > 0 else 0

    return {
        "total_entries": total_entries,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
        "most_used_entries": most_used_list,
        "estimated_api_calls_saved": total_hits - total_entries,
    }

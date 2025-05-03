from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import or_
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from database.session import get_db
from auth.middleware import require_admin
from database.models import (
    User,
    Text,
    Chunk,
    ActiveAssessment,
    Completion,
    CompletionQuestion,
    QuestionCache,
    SimplifiedChunk,
)

# Define which tables are manageable through the admin interface
MANAGEABLE_TABLES = {
    "completions": Completion,
    "active_assessments": ActiveAssessment,
    "question_cache": QuestionCache,
    "simplified_chunks": SimplifiedChunk,
    "completion_questions": CompletionQuestion,
}


# Pydantic models for request/response validation
class TableInfo(BaseModel):
    name: str
    display_name: str
    description: str


class RecordResponse(BaseModel):
    id: str
    data: Dict[str, Any]
    context: Dict[str, Any] = {}
    is_deleted: Optional[bool] = None


class RecordListResponse(BaseModel):
    total: int
    records: List[RecordResponse]


# Create router
router = APIRouter(prefix="/database", tags=["Admin Database Management"])


@router.get("/tables", response_model=List[TableInfo])
async def get_manageable_tables(
    db: Session = Depends(get_db), current_user: User = Depends(require_admin)
):
    """Get list of tables that can be managed through admin interface"""
    tables = [
        TableInfo(
            name="completions",
            display_name="Completions",
            description="Student test completion records",
        ),
        TableInfo(
            name="active_assessments",
            display_name="Active Assessments",
            description="In-progress reading assessments",
        ),
        TableInfo(
            name="question_cache",
            display_name="Question Cache",
            description="Cached AI-generated questions",
        ),
        TableInfo(
            name="simplified_chunks",
            display_name="Simplified Text Chunks",
            description="Simplified versions of text chunks",
        ),
        TableInfo(
            name="completion_questions",
            display_name="Completion Questions",
            description="Questions from completion tests",
        ),
    ]

    return tables


@router.get("/{table_name}", response_model=RecordListResponse)
async def get_table_records(
    table_name: str = Path(..., description="Name of the table to query"),
    search: Optional[str] = Query(
        None, description="Search term for filtering records"
    ),
    include_deleted: bool = Query(False, description="Include soft-deleted records"),
    limit: int = Query(50, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get records from a specific table with filtering options"""
    if table_name not in MANAGEABLE_TABLES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' not found or not manageable",
        )

    # Get model class for the requested table
    model_class = MANAGEABLE_TABLES[table_name]

    try:
        # Start building query
        query = db.query(model_class)

        # Filter by soft delete status if applicable
        if hasattr(model_class, "is_deleted") and not include_deleted:
            query = query.filter(model_class.is_deleted == False)

        # Apply search if provided
        if search:
            # Implement dynamic search based on table type
            if table_name == "completions" or table_name == "active_assessments":
                # Join with user table to search by student name
                query = query.join(User, model_class.student_id == User.id).filter(
                    or_(
                        User.full_name.ilike(f"%{search}%"),
                        User.email.ilike(f"%{search}%"),
                    )
                )
            elif hasattr(model_class, "title"):
                query = query.filter(model_class.title.ilike(f"%{search}%"))
            elif hasattr(model_class, "content"):
                query = query.filter(model_class.content.ilike(f"%{search}%"))

        # Get total count
        total = query.count()

        # Apply pagination
        records = query.limit(limit).offset(offset).all()

        # Format response with context information
        result = []
        for record in records:
            # Convert record to dictionary, excluding SQLAlchemy relationships
            record_dict = {
                c.name: getattr(record, c.name) for c in record.__table__.columns
            }

            # Add context information based on table type
            context = {}

            if table_name in ["completions", "active_assessments"]:
                # Get student information
                student = db.query(User).filter(User.id == record.student_id).first()
                if student:
                    context["student_name"] = student.full_name
                    context["student_email"] = student.email

                # Get text information
                text = db.query(Text).filter(Text.id == record.text_id).first()
                if text:
                    context["text_title"] = text.title
                    context["text_grade"] = text.grade_level

                # For active assessments - get current chunk
                if table_name == "active_assessments" and record.current_chunk_id:
                    chunk = (
                        db.query(Chunk)
                        .filter(Chunk.id == record.current_chunk_id)
                        .first()
                    )
                    if chunk:
                        context["current_chunk_preview"] = (
                            chunk.content[:100] + "..."
                            if len(chunk.content) > 100
                            else chunk.content
                        )

            elif table_name == "question_cache":
                # Get chunk information
                chunk = db.query(Chunk).filter(Chunk.id == record.chunk_id).first()
                if chunk:
                    text = db.query(Text).filter(Text.id == chunk.text_id).first()
                    if text:
                        context["text_title"] = text.title
                        context["chunk_preview"] = (
                            chunk.content[:100] + "..."
                            if len(chunk.content) > 100
                            else chunk.content
                        )

            elif table_name == "simplified_chunks":
                # Get original chunk
                chunk = db.query(Chunk).filter(Chunk.id == record.chunk_id).first()
                if chunk:
                    text = db.query(Text).filter(Text.id == chunk.text_id).first()
                    if text:
                        context["text_title"] = text.title
                        context["original_content"] = (
                            chunk.content[:100] + "..."
                            if len(chunk.content) > 100
                            else chunk.content
                        )

            elif table_name == "completion_questions":
                # Get completion information
                completion = (
                    db.query(Completion)
                    .filter(Completion.id == record.completion_id)
                    .first()
                )
                if completion:
                    student = (
                        db.query(User).filter(User.id == completion.student_id).first()
                    )
                    text = db.query(Text).filter(Text.id == completion.text_id).first()
                    if student and text:
                        context["student_name"] = student.full_name
                        context["text_title"] = text.title

            result.append(
                RecordResponse(
                    id=record.id,
                    data=record_dict,
                    context=context,
                    is_deleted=record_dict.get("is_deleted"),
                )
            )

        return RecordListResponse(total=total, records=result)

    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@router.delete("/{table_name}/{record_id}")
async def delete_record(
    table_name: str = Path(..., description="Name of the table"),
    record_id: str = Path(..., description="ID of the record to delete"),
    hard_delete: bool = Query(
        False, description="Perform hard delete instead of soft delete"
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Delete a record from the specified table"""
    if table_name not in MANAGEABLE_TABLES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' not found or not manageable",
        )

    # Get model class for the requested table
    model_class = MANAGEABLE_TABLES[table_name]  # This needs to be here!

    try:
        # Find the record
        record = db.query(model_class).filter(model_class.id == record_id).first()

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Record with ID {record_id} not found in {table_name}",
            )

        # Handle special case for completions with hard delete
        if table_name == "completions" and hard_delete:
            from datetime import datetime, timezone  # Import datetime for timestamp

            # Store the necessary info before deleting
            student_id = record.student_id
            text_id = record.text_id
            assessment_id = record.assessment_id
            final_test_level = record.final_test_level
            final_test_difficulty = record.final_test_difficulty

            # Delete associated questions
            db.query(CompletionQuestion).filter(
                CompletionQuestion.completion_id == record_id
            ).delete()

            # Delete the record
            db.delete(record)

            # Create a new pending completion record
            new_completion = Completion(
                student_id=student_id,
                text_id=text_id,
                assessment_id=assessment_id,
                final_test_level=final_test_level,
                final_test_difficulty=final_test_difficulty,
                test_status="pending",
                completion_triggered_at=datetime.now(timezone.utc),
            )

            db.add(new_completion)
            db.commit()

            return {
                "message": f"Record {record_id} successfully deleted and reset for retake. New completion ID: {new_completion.id}"
            }

        # Handle regular delete cases
        if hard_delete:
            # Hard delete
            db.delete(record)
        else:
            # Soft delete if supported
            if hasattr(record, "soft_delete"):
                record.soft_delete()
            else:
                # Fall back to hard delete if soft delete not supported
                db.delete(record)

        db.commit()

        return {"message": f"Record {record_id} successfully deleted from {table_name}"}

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@router.put("/{table_name}/{record_id}/restore")
async def restore_record(
    table_name: str = Path(..., description="Name of the table"),
    record_id: str = Path(..., description="ID of the record to restore"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Restore a soft-deleted record"""
    if table_name not in MANAGEABLE_TABLES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' not found or not manageable",
        )

    model_class = MANAGEABLE_TABLES[table_name]

    # Check if model supports soft delete
    if not hasattr(model_class, "is_deleted"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Table '{table_name}' does not support soft deletion/restoration",
        )

    try:
        # Find the record (including deleted ones)
        record = db.query(model_class).filter(model_class.id == record_id).first()

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Record with ID {record_id} not found in {table_name}",
            )

        # Restore the record
        record.is_deleted = False
        db.commit()

        return {"message": f"Record {record_id} successfully restored in {table_name}"}

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )

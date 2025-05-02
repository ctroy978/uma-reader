# routers/student/completion_test.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel
import google.generativeai as genai
from google.genai import types
import os
from sqlalchemy import and_
import uuid

from database.session import get_db
from database.models import (
    Completion,
    CompletionQuestion,
    Text,
    Chunk,
    User,
    QuestionCategory,
)
from auth.middleware import require_user

router = APIRouter(tags=["completion-test"])

# Set up the Google API key
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


# Define request and response models
class AnswerSubmission(BaseModel):
    answer: str


class QuestionResponse(BaseModel):
    id: str
    question_text: str
    category: str
    is_last: bool
    progress: int
    total_questions: int

    class Config:
        from_attributes = True


class TestSummary(BaseModel):
    overall_score: float
    correct_answers: int
    total_questions: int
    category_scores: dict
    completion_id: str

    class Config:
        from_attributes = True


# At the top of routers/student/completion_test.py, right after creating the router

router = APIRouter(tags=["completion-tests"])


async def reset_test_questions(completion_id: str, db: Session):
    """Helper function to delete all existing questions for a test"""
    try:
        # Find and delete all existing questions
        questions = (
            db.query(CompletionQuestion)
            .filter(CompletionQuestion.completion_id == completion_id)
            .all()
        )

        for question in questions:
            db.delete(question)

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error resetting test questions: {str(e)}")
        return False


@router.post("/{completion_id}/initialize", response_model=dict)
async def initialize_completion_test(
    completion_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Initialize the completion test by generating questions and caching the text content"""
    print("DEBUG: Initialize endpoint called with ID:", completion_id)

    try:
        # Verify completion record exists and belongs to the user
        completion = (
            db.query(Completion)
            .filter(
                Completion.id == completion_id,
                Completion.student_id == user.id,
                Completion.test_status == "in_progress",
                Completion.is_deleted == False,
            )
            .first()
        )

        print(f"DEBUG: Completion query result: {completion}")

        if not completion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Completion test not found or not in progress",
            )

        # Reset any existing questions to start fresh
        await reset_test_questions(completion_id, db)
        print("DEBUG: Reset existing questions for a fresh start")

        # Fetch the full text content
        print("DEBUG: Fetching text content")
        text = db.query(Text).filter(Text.id == completion.text_id).first()

        print(f"DEBUG: Text found: {text}")

        if not text:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Text not found",
            )

        # Get all chunks in proper order
        print("DEBUG: Retrieving text chunks")
        chunks = []
        # Start with the first chunk
        first_chunk = (
            db.query(Chunk)
            .filter(
                Chunk.text_id == text.id,
                Chunk.is_first == True,
                Chunk.is_deleted == False,
            )
            .first()
        )

        print(f"DEBUG: First chunk found: {first_chunk}")

        if not first_chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Text has no content",
            )

        # Follow the chain of chunks
        current_chunk = first_chunk
        while current_chunk:
            chunks.append(current_chunk)
            if current_chunk.next_chunk_id:
                current_chunk = db.query(Chunk).get(current_chunk.next_chunk_id)
            else:
                current_chunk = None

        print(f"DEBUG: Found {len(chunks)} chunks")

        # Combine all chunks into a single text
        full_text = " ".join([chunk.content for chunk in chunks])
        print(f"DEBUG: Combined text length: {len(full_text)} characters")

        # Get categories for the questions
        print("DEBUG: Retrieving question categories")
        categories = (
            db.query(QuestionCategory)
            .order_by(QuestionCategory.progression_order)
            .all()
        )

        print(f"DEBUG: Found {len(categories)} categories")

        if not categories:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No question categories found in the database",
            )

        category_names = [category.category_name for category in categories]
        print(f"DEBUG: Category names: {category_names}")

        # Determine the number of questions (between 5 and 10)
        question_count = min(10, 5 + max(0, len(chunks) - 3))
        print(f"DEBUG: Will generate {question_count} questions")

        # Check if API key is configured
        api_key = os.getenv("GEMINI_API_KEY")
        print(f"DEBUG: API key found: {'Yes' if api_key else 'No'}")

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google API key not configured",
            )

        # Initialize model
        print("DEBUG: Initializing Gemini model")
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash-001")
            print("DEBUG: Model initialized successfully")
        except Exception as e:
            print(f"DEBUG ERROR: Model initialization failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error initializing Gemini model: {str(e)}",
            )

        # Create a system prompt with clear formatting instructions
        system_prompt = f"""
        You are an expert reading assessment system. 
        Based on the text provided, generate {question_count} reading comprehension questions.
        
        The questions should:
        1. Test understanding at different levels of complexity
        2. Cover different parts of the text
        3. Include a mix of question categories: {', '.join(category_names)}
        4. Be appropriate for grade level {text.grade_level}
        
        IMPORTANT: Format your response as a properly formatted JSON array of objects.
        Each question object MUST include these exact field names:
        - "question_text": The actual question (string)
        - "category": One of the categories listed above (string)
        - "difficulty": Either "basic", "intermediate", or "advanced" (string)
        
        Example of correct format:
        [
          {{
            "question_text": "What is the main character's name?",
            "category": "literal_basic",
            "difficulty": "basic"
          }},
          {{
            "question_text": "Why did the author use repetition in paragraph 3?",
            "category": "structural_advanced", 
            "difficulty": "advanced"
          }}
        ]
        
        Return ONLY valid JSON. Do not include any explanations or text outside the JSON array.
        """

        print("DEBUG: Created system prompt")

        # Attempt to generate and parse questions with retries
        max_attempts = 3
        questions_json = None
        last_error = None
        last_response_text = None

        for attempt in range(1, max_attempts + 1):
            print(f"DEBUG: Attempt {attempt} to generate questions")
            try:
                # Generate questions with Gemini API
                response = model.generate_content(
                    [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": f"{system_prompt}\n\nText to analyze:\n{full_text}"
                                }
                            ],
                        },
                    ]
                )

                response_text = response.text
                last_response_text = response_text
                print(
                    f"DEBUG: Received response from Gemini API (Length: {len(response_text)} chars)"
                )

                # Parse JSON from response
                # Try to extract JSON from markdown codeblocks if present
                import json
                import re

                # Look for JSON in code blocks
                json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                match = re.search(json_pattern, response_text)

                if match:
                    # If we found a code block, extract the content
                    json_content = match.group(1).strip()
                    print(f"DEBUG: Extracted JSON from code block")
                else:
                    # Otherwise use the raw text
                    json_content = response_text.strip()
                    print(f"DEBUG: Using raw text as JSON")

                # Clean up common formatting issues
                json_content = json_content.replace("questin_text", "question_text")
                json_content = json_content.replace("catergory", "category")
                json_content = json_content.replace("diffculty", "difficulty")
                json_content = json_content.replace("difficutly", "difficulty")

                # Try parsing the cleaned JSON
                questions_json = json.loads(json_content)
                print(
                    f"DEBUG: Successfully parsed JSON with {len(questions_json)} questions"
                )

                # Validate parsed JSON
                if not isinstance(questions_json, list) or len(questions_json) == 0:
                    raise ValueError("Response is not a valid list of questions")

                # Validate each question has required fields
                for idx, q in enumerate(questions_json):
                    if not isinstance(q, dict):
                        raise ValueError(f"Question {idx} is not a valid object")

                    # Check required fields
                    required_fields = ["question_text", "category", "difficulty"]
                    for field in required_fields:
                        if field not in q:
                            raise ValueError(
                                f"Question {idx} is missing required field: {field}"
                            )

                # If we get here, JSON is valid
                break

            except Exception as e:
                last_error = str(e)
                print(f"DEBUG ERROR: Attempt {attempt} - {last_error}")

                # If not the last attempt, try again with stronger formatting instructions
                if attempt < max_attempts:
                    system_prompt += f"""
                    
                    IMPORTANT: Your previous response had formatting issues. Please ensure:
                    1. You return ONLY valid JSON
                    2. All field names are exact: "question_text", "category", "difficulty"
                    3. No additional explanatory text
                    4. No markdown formatting except for the JSON itself
                    """

        # If we still don't have valid questions after max attempts
        if questions_json is None:
            error_message = (
                f"Failed to generate valid questions after {max_attempts} attempts. "
                f"Last error: {last_error}. Please return to your dashboard and try the completion test again."
            )
            print(f"DEBUG ERROR: Final failure - {error_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_message,
            )

        # Create CompletionQuestion records
        print("DEBUG: Creating question records in database")
        question_ids = []
        prev_question_id = None

        for i, q in enumerate(questions_json):
            # Create the question record
            question = CompletionQuestion(
                id=str(uuid.uuid4()),
                completion_id=completion_id,
                category=q["category"],
                difficulty=q["difficulty"],
                question_text=q["question_text"],
                is_answered=False,
            )

            # Link to previous question if it exists
            if prev_question_id:
                # Find the previous question and set its next_question_id
                prev_question = (
                    db.query(CompletionQuestion)
                    .filter(CompletionQuestion.id == prev_question_id)
                    .first()
                )
                if prev_question:
                    prev_question.next_question_id = question.id

            db.add(question)
            question_ids.append(question.id)
            prev_question_id = question.id

            print(f"DEBUG: Created question {i+1} with ID {question.id}")

        # Commit the questions to the database
        print("DEBUG: Committing questions to database")
        db.commit()
        print("DEBUG: Commit successful")

        return {
            "message": "Completion test initialized successfully",
            "completion_id": completion_id,
            "questions_count": len(question_ids),
            "first_question_id": question_ids[0] if question_ids else None,
        }

    except HTTPException as he:
        print(f"DEBUG ERROR: HTTP Exception: {he.detail}")
        db.rollback()
        raise
    except Exception as e:
        print(f"DEBUG ERROR: Unexpected exception: {str(e)}")
        print(f"DEBUG ERROR: Exception type: {type(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating questions: {str(e)}",
        )


@router.post("/{completion_id}/question/{question_id}/answer", response_model=dict)
async def submit_answer(
    completion_id: str,
    question_id: str,
    answer: AnswerSubmission,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Submit an answer to a completion test question"""
    # Verify completion record exists and belongs to the user
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.test_status == "in_progress",
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found or not in progress",
        )

    # Get the question
    question = (
        db.query(CompletionQuestion)
        .filter(
            CompletionQuestion.id == question_id,
            CompletionQuestion.completion_id == completion_id,
        )
        .first()
    )

    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    if question.is_answered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question already answered",
        )

    # Record the student's answer
    question.student_answer = answer.answer
    question.is_answered = True

    # For now, we won't evaluate correctness here
    # This will be done in a bulk operation when finalizing the test

    db.commit()

    return {
        "message": "Answer submitted successfully",
        "next_question_id": question.next_question_id,
    }


@router.post("/{completion_id}/finalize", response_model=TestSummary)
async def finalize_completion_test(
    completion_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Finalize the completion test and calculate scores"""
    # Verify completion record exists and belongs to the user
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.test_status == "in_progress",
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found or not in progress",
        )

    # Get all questions for this completion
    questions = (
        db.query(CompletionQuestion)
        .filter(CompletionQuestion.completion_id == completion_id)
        .all()
    )

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No questions found for this test",
        )

    # Check if all questions have been answered
    unanswered = any(not q.is_answered for q in questions)
    if unanswered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not all questions have been answered",
        )

    # Evaluate answers using AI
    try:
        # Get text content for context
        text = db.query(Text).filter(Text.id == completion.text_id).first()

        # Fetch chunks and combine into full text
        chunks = []
        first_chunk = (
            db.query(Chunk)
            .filter(
                Chunk.text_id == text.id,
                Chunk.is_first == True,
                Chunk.is_deleted == False,
            )
            .first()
        )

        current_chunk = first_chunk
        while current_chunk:
            chunks.append(current_chunk)
            if current_chunk.next_chunk_id:
                current_chunk = db.query(Chunk).get(current_chunk.next_chunk_id)
            else:
                current_chunk = None

        full_text = " ".join([chunk.content for chunk in chunks])

        # Initialize the Gemini model
        model_name = "gemini-1.5-flash-001"
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        system_instruction = """
        You are an expert reading assessment evaluator. 
        Evaluate if student answers to questions are correct based on the provided text.

        Consider:
        1. The answer's factual accuracy compared to the text
        2. The completeness of the answer
        3. The student's demonstration of understanding
        """

        # Keep track of results by category
        category_results = {}
        correct_count = 0
        invalid_responses = []

        # Use the model to evaluate each question (no caching)
        for question in questions:
            eval_prompt = f"""
            {system_instruction}

            TEXT: {full_text}
            
            QUESTION: {question.question_text}
            
            STUDENT ANSWER: {question.student_answer}
            
            Is the answer correct? Return ONLY "true" if the answer is correct or "false" if incorrect.
            """

            try:
                response = model.generate_content(eval_prompt)
                # Extract text from response.
                if response.candidates and response.candidates[0].content.parts:
                    is_correct_text = (
                        response.candidates[0].content.parts[0].text.strip().lower()
                    )
                    # validation moved here
                    if is_correct_text not in ["true", "false"]:
                        invalid_responses.append(
                            {
                                "question_id": question.id,
                                "response_text": is_correct_text,
                            }
                        )
                        is_correct = False
                    else:
                        is_correct = is_correct_text == "true"
                else:
                    raise Exception("Response is missing content")
            except Exception as e:
                raise Exception(f"Error while generating the answer {str(e)}")

            # Update the question record
            question.is_correct = is_correct

            if is_correct:
                correct_count += 1

            # Track category results
            if question.category not in category_results:
                category_results[question.category] = {
                    "total": 0,
                    "correct": 0,
                }

            category_results[question.category]["total"] += 1
            if is_correct:
                category_results[question.category]["correct"] += 1

        # check if we had any invalid responses
        if len(invalid_responses) > 0:
            # we have invalid responses, log them
            for response in invalid_responses:
                print(
                    f"ERROR: Invalid response for question id: {response['question_id']}, response: {response['response_text']}"
                )
            raise Exception(
                f"Invalid response from model on {len(invalid_responses)} question(s)"
            )

        # Calculate category success percentages
        category_scores = {}
        for category, results in category_results.items():
            if results["total"] > 0:
                category_scores[category] = (
                    results["correct"] / results["total"]
                ) * 100
            else:
                category_scores[category] = 0

        # Update completion record with results
        completion.test_status = "completed"
        completion.completed_at = datetime.now(timezone.utc)
        completion.overall_score = (correct_count / len(questions)) * 100
        completion.total_questions = len(questions)
        completion.correct_answers = correct_count

        # Update category-specific scores
        for category, score in category_scores.items():
            setattr(completion, f"{category.lower()}_success", score)

        db.commit()

        return TestSummary(
            overall_score=completion.overall_score,
            correct_answers=correct_count,
            total_questions=len(questions),
            category_scores=category_scores,
            completion_id=completion_id,
        )

    except Exception as e:
        db.rollback()
        print(f"ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error evaluating answers: {str(e)}",
        )


@router.get("/{completion_id}/first-question", response_model=dict)
async def get_first_question(
    completion_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get the first question of a completion test"""
    # Verify completion record exists and belongs to the user
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.test_status == "in_progress",
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found or not in progress",
        )

    # Find the first question (one with no previous question pointing to it)
    # This is a bit inefficient but works for a small dataset
    all_questions = (
        db.query(CompletionQuestion)
        .filter(CompletionQuestion.completion_id == completion_id)
        .all()
    )

    if not all_questions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No questions found for this test",
        )

    # Find all question IDs that are referenced as next_question_id
    next_ids = set()
    for q in all_questions:
        if q.next_question_id:
            next_ids.add(q.next_question_id)

    # The first question is the one that isn't in the next_ids set
    first_question = next((q for q in all_questions if q.id not in next_ids), None)

    if not first_question:
        # Fallback to the first question in the list
        first_question = all_questions[0]

    return {
        "first_question_id": first_question.id,
        "total_questions": len(all_questions),
    }


@router.get("/{completion_id}/question/{question_id}", response_model=QuestionResponse)
async def get_question(
    completion_id: str,
    question_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get a specific question from a completion test"""
    # Verify completion record exists and belongs to the user
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.test_status == "in_progress",
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found or not in progress",
        )

    # Get the requested question
    question = (
        db.query(CompletionQuestion)
        .filter(
            CompletionQuestion.id == question_id,
            CompletionQuestion.completion_id == completion_id,
        )
        .first()
    )

    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found",
        )

    # Count total questions and current progress
    total_questions = (
        db.query(CompletionQuestion)
        .filter(CompletionQuestion.completion_id == completion_id)
        .count()
    )

    # FIX: Determine if this is the last question by explicitly checking next_question_id
    is_last = question.next_question_id is None

    # Count how many questions come before this one to determine progress
    # This is a bit inefficient but works for small question sets
    progress = 1  # Start at 1 for the current question

    # Follow previous links to count questions that come before
    current_id = question_id
    seen_ids = set([current_id])

    while True:
        # Find any question that has this as its next_question_id
        prev_question = (
            db.query(CompletionQuestion)
            .filter(
                CompletionQuestion.completion_id == completion_id,
                CompletionQuestion.next_question_id == current_id,
            )
            .first()
        )

        if prev_question and prev_question.id not in seen_ids:
            progress += 1
            current_id = prev_question.id
            seen_ids.add(current_id)
        else:
            break

    return QuestionResponse(
        id=question.id,
        question_text=question.question_text,
        category=question.category,
        is_last=is_last,  # FIX: This now correctly indicates if it's the last question
        progress=progress,
        total_questions=total_questions,
    )


@router.get("/{completion_id}/unanswered-questions")
async def get_unanswered_questions(
    completion_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get all unanswered questions for a completion test"""
    # Verify completion record exists and belongs to the user
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.test_status == "in_progress",
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found or not in progress",
        )

    # Get all unanswered questions
    unanswered_questions = (
        db.query(CompletionQuestion)
        .filter(
            CompletionQuestion.completion_id == completion_id,
            CompletionQuestion.is_answered == False,
        )
        .all()
    )

    # Count total questions
    total_questions = (
        db.query(CompletionQuestion)
        .filter(CompletionQuestion.completion_id == completion_id)
        .count()
    )

    return {
        "questions": [
            {
                "id": q.id,
                "question_text": q.question_text,
                "category": q.category,
            }
            for q in unanswered_questions
        ],
        "total_questions": total_questions,
        "unanswered_count": len(unanswered_questions),
    }


@router.get("/{completion_id}/next-question")
async def get_next_question(
    completion_id: str,
    current: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get the next question in the chain after the current question"""
    # Verify completion record exists and belongs to the user
    completion = (
        db.query(Completion)
        .filter(
            Completion.id == completion_id,
            Completion.student_id == user.id,
            Completion.test_status == "in_progress",
            Completion.is_deleted == False,
        )
        .first()
    )

    if not completion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completion test not found or not in progress",
        )

    # Get the current question
    current_question = (
        db.query(CompletionQuestion)
        .filter(
            CompletionQuestion.id == current,
            CompletionQuestion.completion_id == completion_id,
        )
        .first()
    )

    if not current_question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current question not found",
        )

    # Check if there's a next question in the chain
    if current_question.next_question_id:
        next_question = db.query(CompletionQuestion).get(
            current_question.next_question_id
        )

        if next_question:
            # Determine if this is the last question (no next_question_id)
            is_last = next_question.next_question_id is None

            # Count total questions
            total_questions = (
                db.query(CompletionQuestion)
                .filter(CompletionQuestion.completion_id == completion_id)
                .count()
            )

            # Count current progress
            # For simplicity, we'll count how many questions come before this one
            previous_questions = []
            current_id = next_question.id
            seen_ids = set([current_id])

            while True:
                # Find any question that has this as its next_question_id
                prev_question = (
                    db.query(CompletionQuestion)
                    .filter(
                        CompletionQuestion.completion_id == completion_id,
                        CompletionQuestion.next_question_id == current_id,
                    )
                    .first()
                )

                if prev_question and prev_question.id not in seen_ids:
                    previous_questions.append(prev_question)
                    current_id = prev_question.id
                    seen_ids.add(current_id)
                else:
                    break

            # Progress is the number of questions that come before + 1 (for the current question)
            progress = len(previous_questions) + 1

            return {
                "next_question_id": next_question.id,
                "question_text": next_question.question_text,
                "category": next_question.category,
                "is_last": is_last,
                "progress": progress,
                "total_questions": total_questions,
            }

    # If we get here, there's no next question
    return {"next_question_id": None}


# Add this endpoint to routers/student/completion_test.py


@router.get("/{completion_id}/text-chunks", response_model=dict)
async def get_completion_text_chunks(
    completion_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    """Get all text chunks for a completion test"""
    try:
        # Verify completion record exists and belongs to the user
        completion = (
            db.query(Completion)
            .filter(
                Completion.id == completion_id,
                Completion.student_id == user.id,
                Completion.is_deleted == False,
            )
            .first()
        )

        if not completion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Completion test not found",
            )

        # Get the text
        text = db.query(Text).filter(Text.id == completion.text_id).first()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Text not found",
            )

        # Get all chunks in order
        chunks = []
        first_chunk = (
            db.query(Chunk)
            .filter(
                Chunk.text_id == text.id,
                Chunk.is_first == True,
                Chunk.is_deleted == False,
            )
            .first()
        )

        if not first_chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Text has no content",
            )

        # Follow the chain of chunks
        current_chunk = first_chunk
        chunk_index = 1
        while current_chunk:
            chunks.append(
                {
                    "id": current_chunk.id,
                    "content": current_chunk.content,
                    "index": chunk_index,
                }
            )
            chunk_index += 1

            if current_chunk.next_chunk_id:
                current_chunk = db.query(Chunk).get(current_chunk.next_chunk_id)
            else:
                current_chunk = None

        return {"text_title": text.title, "chunks": chunks}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log and convert other exceptions to 500 errors
        print(f"Error fetching text chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching text chunks: {str(e)}",
        )

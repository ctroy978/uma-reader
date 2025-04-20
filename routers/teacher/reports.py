# app/routers/teacher/reports.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv

from database.session import get_db
from database.models import User, Text, Completion, CompletionQuestion
from auth.middleware import require_user

# Load environment variables and initialize AI model
load_dotenv()
model = GeminiModel("gemini-2.0-flash")

router = APIRouter(tags=["teacher-reports"])

# =============================================================================
# CONFIGURATION FLAGS
# =============================================================================
# This flag controls whether cumulative reports are available to users.
# It is currently set to False to disable cumulative reports temporarily
# until front-end issues are resolved. When the front-end is fixed,
# set this to True to re-enable the functionality.
#
# IMPORTANT: When this flag is False, all cumulative report endpoints will
# return appropriate error messages, and no cumulative reports will be
# included in the list of available reports.
# =============================================================================
CUMULATIVE_REPORTS_ENABLED = False  # Temporarily disable cumulative reports

MIN_NUMBER_COMPLETIONS = 3


# Pydantic models for AI responses
class SingleTestAnalysis(BaseModel):
    """Model for AI-generated single test analysis"""

    reading_level_analysis: str = Field(
        ..., description="Analysis of the student's current reading level and skills"
    )
    text_specific_insights: str = Field(
        ..., description="Insights related to performance with this text type and genre"
    )
    recommended_activities: List[str] = Field(
        ..., description="2-3 targeted activities to improve reading comprehension"
    )


class CumulativeAnalysis(BaseModel):
    """Model for AI-generated cumulative analysis"""

    progression_analysis: str = Field(
        ..., description="Analysis of reading level progression over time"
    )
    text_performance_insights: str = Field(
        ..., description="Insights about performance across text types and genres"
    )
    development_recommendations: str = Field(
        ..., description="Pattern-based recommendations for continued development"
    )
    teaching_strategies: List[str] = Field(
        ..., description="3-4 evidence-based teaching strategies for specific needs"
    )


# Response models for graph data
class CategoryPerformance(BaseModel):
    category: str
    score: float
    attempts: int


class SingleTestGraphData(BaseModel):
    categories: List[CategoryPerformance]
    text_metadata: Dict[str, Any]


class ProgressionPoint(BaseModel):
    date: datetime
    level: str
    score: float
    text_id: str
    text_title: str


class CumulativeGraphData(BaseModel):
    progression: List[ProgressionPoint]
    category_trends: Dict[str, List[float]]
    text_performance: Dict[str, Dict[str, List[float]]]


# Ensure teacher role
def require_teacher(user: User = Depends(require_user)):
    """Middleware to ensure the user is a teacher"""
    if user.role_name != "TEACHER" and user.role_name != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers can access this resource",
        )
    return user


async def generate_single_test_analysis(
    completion: Completion, text: Text
) -> SingleTestAnalysis:
    """Generate analysis for a single test using AI"""
    try:
        agent = Agent(
            model=model,
            result_type=SingleTestAnalysis,
            system_prompt="You are an expert reading teacher analyzing student assessment results.",
        )

        # Extract categories with non-zero scores
        categories = {}
        for category in [
            "literal_basic",
            "literal_detailed",
            "vocabulary",
            "inferential_simple",
            "inferential_complex",
            "structural_basic",
            "structural_advanced",
        ]:
            score = getattr(completion, f"{category}_success", 0)
            if score > 0:
                categories[category] = score

        # Determine strengths and areas for development
        strengths = [k for k, v in categories.items() if v >= 75.0]
        development_areas = [k for k, v in categories.items() if 0 < v < 75.0]

        genre_names = (
            [g.genre_name for g in text.genres] if hasattr(text, "genres") else []
        )

        prompt = f"""
        Analyze this student's reading assessment performance:
        
        Overall score: {completion.overall_score}%
        Final test level: {completion.final_test_level}
        
        Text information:
        - Title: {text.title}
        - Grade level: {text.grade_level}
        - Type: {text.type_name}
        - Form: {text.form_name}
        - Genres: {', '.join(genre_names)}
        
        Performance by category:
        {categories}
        
        Identified strengths: {', '.join(strengths)}
        Areas for development: {', '.join(development_areas)}
        
        Based on this single assessment, provide:
        1. A concise analysis of the student's current reading level and skills
        2. Specific insights related to their performance with this particular text type and genre
        3. 2-3 targeted activities that would help them improve their reading comprehension
        
        Format your response as a structured analysis with clear sections.
        """

        result = await agent.run(prompt)

        if not result or not result.data:
            raise ValueError("Failed to generate analysis")

        return result.data

    except Exception as e:
        print(f"AI Analysis Generation Error: {str(e)}")
        # Provide fallback response if AI fails
        return SingleTestAnalysis(
            reading_level_analysis="Unable to generate detailed analysis. The student completed the assessment with some strengths in reading comprehension.",
            text_specific_insights="Review the score breakdown for insights on performance with this text.",
            recommended_activities=[
                "Re-read passages and identify main ideas",
                "Practice summarizing content in own words",
            ],
        )


async def generate_cumulative_analysis(
    completions: List[Completion], texts: List[Text]
) -> CumulativeAnalysis:
    """Generate cumulative analysis using AI"""
    try:
        agent = Agent(
            model=model,
            result_type=CumulativeAnalysis,
            system_prompt="You are an expert reading teacher analyzing long-term student assessment results.",
        )

        # Create a mapping of text_id to text for easy lookup
        text_map = {text.id: text for text in texts if text is not None}

        # Extract trend data
        scores_over_time = [
            (c.completed_at.isoformat(), c.overall_score) for c in completions
        ]
        levels_over_time = [
            (c.completed_at.isoformat(), c.final_test_level) for c in completions
        ]

        # Analyze consistent strengths and weaknesses
        categories = [
            "literal_basic",
            "literal_detailed",
            "vocabulary",
            "inferential_simple",
            "inferential_complex",
            "structural_basic",
            "structural_advanced",
        ]

        # Calculate average score for each category
        category_avgs = {}
        for category in categories:
            scores = [getattr(c, f"{category}_success", 0) for c in completions]
            valid_scores = [s for s in scores if s > 0]
            if valid_scores:
                category_avgs[category] = sum(valid_scores) / len(valid_scores)

        # Identify consistent strengths and weaknesses
        consistent_strengths = [k for k, v in category_avgs.items() if v >= 75.0]
        consistent_weaknesses = [k for k, v in category_avgs.items() if 0 < v < 75.0]

        # Analyze text performance by type
        text_types = {}
        for c in completions:
            text = text_map.get(c.text_id)
            if not text:
                continue

            if text.type_name not in text_types:
                text_types[text.type_name] = []
            text_types[text.type_name].append(c.overall_score)

        # Format the text types performance
        text_performance = {}
        for type_name, scores in text_types.items():
            text_performance[type_name] = sum(scores) / len(scores)

        prompt = f"""
        Analyze this student's reading development across {len(completions)} assessments:
        
        Score progression: {scores_over_time}
        Reading level progression: {levels_over_time}
        
        Consistent strengths: {', '.join(consistent_strengths)}
        Consistent areas for development: {', '.join(consistent_weaknesses)}
        
        Text performance patterns:
        {text_performance}
        
        Based on this cumulative assessment history, provide:
        1. An analysis of the student's reading level progression over time
        2. Insights about their performance across different text types and genres
        3. Pattern-based recommendations for continued reading development
        4. 3-4 evidence-based teaching strategies that would address their specific needs
        
        Format your response as a comprehensive analysis with clearly defined sections.
        """

        result = await agent.run(prompt)

        if not result or not result.data:
            raise ValueError("Failed to generate cumulative analysis")

        return result.data

    except Exception as e:
        print(f"AI Cumulative Analysis Generation Error: {str(e)}")
        # Provide fallback response if AI fails
        return CumulativeAnalysis(
            progression_analysis="Unable to generate detailed progression analysis. Review the raw data to observe trends.",
            text_performance_insights="Student has completed multiple assessments across different text types.",
            development_recommendations="Continue to provide varied reading materials and monitor progress.",
            teaching_strategies=[
                "Use guided reading with texts at appropriate levels",
                "Implement comprehension strategy instruction",
                "Provide regular feedback on specific reading skills",
            ],
        )


def analyze_text_performance_for_graph(
    completions: List[Completion], texts: List[Text]
) -> Dict[str, Dict[str, List[float]]]:
    """Analyze student performance across different text attributes"""
    # Create a mapping of text_id to text for easy lookup
    text_map = {text.id: text for text in texts if text is not None}

    # Performance by genre
    genre_scores = {}
    for completion in completions:
        text = text_map.get(completion.text_id)
        if not text or not hasattr(text, "genres"):
            continue

        for genre in text.genres:
            if genre.genre_name not in genre_scores:
                genre_scores[genre.genre_name] = []
            genre_scores[genre.genre_name].append(completion.overall_score)

    # Performance by text type
    type_scores = {}
    for completion in completions:
        text = text_map.get(completion.text_id)
        if not text:
            continue

        if text.type_name not in type_scores:
            type_scores[text.type_name] = []
        type_scores[text.type_name].append(completion.overall_score)

    # Performance by grade level
    grade_scores = {}
    for completion in completions:
        text = text_map.get(completion.text_id)
        if not text:
            continue

        grade_key = f"Grade {text.grade_level}"
        if grade_key not in grade_scores:
            grade_scores[grade_key] = []
        grade_scores[grade_key].append(completion.overall_score)

    return {"by_genre": genre_scores, "by_type": type_scores, "by_grade": grade_scores}


@router.get("")
async def get_all_reports(
    text_title: Optional[str] = Query(None, description="Filter by text title"),
    grade_level: Optional[int] = Query(None, description="Filter by text grade level"),
    date_from: Optional[datetime] = Query(
        None, description="Filter by date range start"
    ),
    date_to: Optional[datetime] = Query(None, description="Filter by date range end"),
    sort_by: Optional[str] = Query("completed_at", description="Field to sort by"),
    sort_order: Optional[str] = Query("desc", description="Sort order (asc or desc)"),
    db: Session = Depends(get_db),
    teacher: User = Depends(require_teacher),
):
    """Get all available reports for a teacher with filtering options

    NOTE: When cumulative reports are disabled via the CUMULATIVE_REPORTS_ENABLED
    configuration flag at the top of this file, this endpoint will only return
    single test reports and not include any cumulative reports in the list.
    """
    try:
        # Base query to find completions for texts created by the current teacher
        query = (
            db.query(Completion, Text, User)
            .join(Text, Completion.text_id == Text.id)
            .join(User, Completion.student_id == User.id)
            .filter(
                Text.teacher_id == teacher.id,  # Filter by teacher_id
                Completion.is_deleted == False,
                Completion.test_status == "completed",
                Text.is_deleted == False,
            )
        )

        # Apply text title filter if provided
        if text_title:
            query = query.filter(Text.title.ilike(f"%{text_title}%"))

        # Apply grade level filter if provided
        if grade_level:
            query = query.filter(Text.grade_level == grade_level)

        # Apply date range filters if provided
        if date_from:
            query = query.filter(Completion.completed_at >= date_from)
        if date_to:
            query = query.filter(Completion.completed_at <= date_to)

        # Apply sorting
        if sort_order.lower() == "asc":
            query = query.order_by(asc(getattr(Completion, sort_by)))
        else:
            query = query.order_by(desc(getattr(Completion, sort_by)))

        # Execute query
        results = query.all()

        # Format response
        reports = []
        for completion, text, student in results:
            # Add single test report
            reports.append(
                {
                    "id": completion.id,
                    "report_type": "single_test",
                    "student_id": student.id,
                    "student_name": student.full_name,
                    "grade_level": text.grade_level,
                    "text_title": text.title,
                    "text_type": text.type_name,
                    "completed_at": completion.completed_at,
                    "overall_score": completion.overall_score,
                }
            )

        # Only add cumulative reports if they are enabled
        if CUMULATIVE_REPORTS_ENABLED:
            # Add cumulative reports for users with enough data
            student_ids = set(r["student_id"] for r in reports)

            for student_id in student_ids:
                student_reports = [r for r in reports if r["student_id"] == student_id]

                if (
                    len(student_reports) >= MIN_NUMBER_COMPLETIONS
                ):  # Minimum for cumulative reports
                    user = db.query(User).filter(User.id == student_id).first()
                    if not user:
                        continue

                    # Calculate date range and average score
                    completion_dates = [r["completed_at"] for r in student_reports]
                    date_range = (max(completion_dates) - min(completion_dates)).days
                    avg_score = sum(r["overall_score"] for r in student_reports) / len(
                        student_reports
                    )

                    reports.append(
                        {
                            "id": f"cumulative-{student_id}",
                            "report_type": "cumulative",
                            "student_id": student_id,
                            "student_name": user.full_name,
                            "grade_level": student_reports[0]["grade_level"],
                            "tests_count": len(student_reports),
                            "days_covered": date_range,
                            "completed_at": max(completion_dates),
                            "overall_score": avg_score,
                        }
                    )

        return reports
    except Exception as e:
        import traceback

        print(f"DEBUG: Exception caught: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching reports: {str(e)}",
        )


@router.get("/student/{student_id}/report/{completion_id}")
async def get_student_single_report(
    student_id: str,
    completion_id: str,
    db: Session = Depends(get_db),
    teacher: User = Depends(require_teacher),
    regenerate: bool = False,
):
    """Generate a single test report with text analysis for a student or teacher"""

    try:
        # Verify the user exists
        user = (
            db.query(User)
            .filter(
                User.id == student_id,
                (User.role_name == "STUDENT") | (User.role_name == "TEACHER"),
                User.is_deleted == False,
            )
            .first()
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Get the completion record
        completion = (
            db.query(Completion)
            .filter(
                Completion.id == completion_id,
                Completion.student_id == student_id,
                Completion.is_deleted == False,
            )
            .first()
        )
        if not completion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Completion record not found",
            )

        # Get the associated text
        text = db.query(Text).filter(Text.id == completion.text_id).first()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Text not found"
            )

        # Verify the teacher has access to this text
        if text.teacher_id != teacher.id and teacher.role_name != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this text's reports",
            )

        # Check if analysis exists and we're not forcing regeneration
        if completion.analysis_content and not regenerate:
            # Use the cached analysis
            analysis = SingleTestAnalysis.parse_raw(completion.analysis_content)
            from_cache = True
        else:
            # Generate a new analysis
            analysis = await generate_single_test_analysis(completion, text)

            # Store the analysis for future use
            completion.analysis_content = analysis.json()
            db.commit()
            from_cache = False

        # Get the questions for this completion
        question_details = []
        for q in completion.questions:
            question_details.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "question_text": q.question_text,
                    "student_answer": q.student_answer,
                    "is_correct": q.is_correct,
                    "time_spent_seconds": q.time_spent_seconds or 0,
                }
            )

        return {
            "student_name": user.full_name,
            "report_type": "single_test",
            "analysis": analysis.dict(),
            "text_title": text.title,
            "text_grade_level": text.grade_level,
            "text_type": text.type_name,
            "completion_date": completion.completed_at,
            "overall_score": completion.overall_score,
            "total_questions": completion.total_questions,
            "correct_answers": completion.correct_answers,
            "from_cache": from_cache,
            "questions": question_details,  # Added question details
        }
    except Exception as e:
        import traceback

        print(f"DEBUG: Exception caught: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating report: {str(e)}",
        )

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
import json

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
CUMULATIVE_REPORTS_ENABLED = True  # Temporarily disable cumulative reports

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


def generate_improved_category_stats(completions):
    """
    Generate improved category statistics with data quality indicators.
    Uses CompletionQuestion records to count actual questions attempted.
    """
    categories = [
        "literal_basic",
        "literal_detailed",
        "vocabulary",
        "inferential_simple",
        "inferential_complex",
        "structural_basic",
        "structural_advanced",
    ]

    # Calculate category stats with quality indicators
    category_stats = {}
    data_quality = {}

    for category in categories:
        scores = []
        attempts_by_completion = []
        total_questions = 0

        for completion in completions:
            # Count actual questions for this category from CompletionQuestion records
            questions_in_category = [
                q
                for q in completion.questions
                if q.category == category and q.is_answered
            ]
            category_attempts = len(questions_in_category)

            # Get the success rate percentage for this category
            score = getattr(completion, f"{category}_success", 0)

            # If there were attempts for this category
            if category_attempts > 0:
                scores.append(score)
                attempts_by_completion.append(category_attempts)
                total_questions += category_attempts

        # Only include categories that have at least one attempt
        if total_questions > 0:
            avg_score = sum(scores) / len(scores) if scores else 0

            category_stats[category] = {
                "score": avg_score,
                "assessments_with_data": len(scores),
                "total_questions": total_questions,
                "avg_questions_per_assessment": (
                    sum(attempts_by_completion) / len(attempts_by_completion)
                    if attempts_by_completion
                    else 0
                ),
            }

            # Add data quality indicator
            data_quality[category] = {
                "limited_data": total_questions < 3,
                "potential_misleading": len(completions)
                > len(scores) * 1.5,  # More than 50% filtered out
                "confidence": (
                    "high"
                    if total_questions >= 10
                    else "medium" if total_questions >= 5 else "low"
                ),
                "coverage_ratio": (
                    len(scores) / len(completions) if len(completions) > 0 else 0
                ),
            }

    return {"stats": category_stats, "data_quality": data_quality}


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


# app/routers/teacher/reports.py
# Replace the entire generate_cumulative_analysis function with this implementation


async def generate_cumulative_analysis(
    completions: List[Completion], texts: List[Text]
) -> CumulativeAnalysis:
    """Generate cumulative analysis using AI with data quality awareness"""
    try:
        agent = Agent(
            model=model,
            result_type=CumulativeAnalysis,
            system_prompt="You are an expert reading teacher analyzing student assessment results. Be cautious about data quality issues and avoid making strong claims when data is limited.",
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

        # Get improved category statistics with data quality indicators
        category_analysis = generate_improved_category_stats(completions)
        category_stats = category_analysis["stats"]
        data_quality = category_analysis["data_quality"]

        # Identify categories with reliable data
        reliable_strengths = []
        reliable_weaknesses = []
        unreliable_categories = []

        for category, stats in category_stats.items():
            # Get the quality indicators for this category
            quality = data_quality.get(category, {})

            # Skip categories with no attempts
            if stats.get("total_questions", 0) == 0:
                continue

            # Categorize based on data quality
            if quality.get("limited_data", True) or quality.get(
                "potential_misleading", False
            ):
                unreliable_categories.append((category, stats["score"]))
            elif stats["score"] >= 75.0:
                reliable_strengths.append(category)
            else:
                reliable_weaknesses.append(category)

        # Analyze text performance by type
        text_types = {}
        text_type_attempts = {}
        for c in completions:
            text = text_map.get(c.text_id)
            if not text:
                continue

            if text.type_name not in text_types:
                text_types[text.type_name] = []
                text_type_attempts[text.type_name] = 0

            text_types[text.type_name].append(c.overall_score)
            text_type_attempts[text.type_name] += 1

        # Format the text types performance with reliability indicators
        text_performance = {}
        for type_name, scores in text_types.items():
            is_reliable = text_type_attempts[type_name] >= 3
            text_performance[type_name] = {
                "average_score": sum(scores) / len(scores),
                "attempts": text_type_attempts[type_name],
                "is_reliable": is_reliable,
            }

        # Add detailed explanation about data interpretation
        data_explanation = """
        IMPORTANT NOTE ON DATA INTERPRETATION:
        
        1. Data Quality Issues:
           - Several categories have limited data (less than 3 question attempts)
           - Some categories show inconsistent coverage across assessments
           - Text type scores are more reliable as they reflect overall performance
        
        2. Interpreting Categories:
           - Category scores only show performance on questions the student attempted
           - A 100% score with very few attempts is not reliable evidence of mastery
           - Categories with no attempts are excluded from this analysis
        
        3. Reliability Guidelines:
           - High reliability: consistent data across multiple assessments with many questions
           - Medium reliability: some data but may have inconsistencies or limited samples
           - Low reliability: very limited data, insufficient for confident conclusions
        """

        prompt = f"""
        Analyze this student's reading development across {len(completions)} assessments, with special attention to data quality issues:
        
        Score progression: {scores_over_time}
        Reading level progression: {levels_over_time}
        
        Performance by category (with data quality insights):
        {category_stats}
        
        Data quality issues:
        {data_quality}
        
        Categories with RELIABLE data:
        - Strengths: {', '.join(reliable_strengths) if reliable_strengths else "None identified with sufficient data"}
        - Areas for development: {', '.join(reliable_weaknesses) if reliable_weaknesses else "None identified with sufficient data"}
        
        Categories with UNRELIABLE data (limited samples or inconsistent coverage):
        {unreliable_categories}
        
        Text performance patterns:
        {text_performance}
        
        {data_explanation}
        
        CRITICAL INSTRUCTIONS:
        1. Focus your analysis ONLY on categories and text types with sufficient reliable data
        2. Explicitly acknowledge data limitations in your analysis
        3. Use cautious language when discussing categories with limited data
        4. If there is insufficient reliable data, state this clearly and focus on providing general guidance
        5. DO NOT make strong claims or specific recommendations based on unreliable data
        6. If apparent contradictions exist (e.g., high category scores but low text performance), explain these as likely data quality issues
        
        Based on this information, provide:
        1. An analysis of the student's reading level progression over time (acknowledging data limitations)
        2. Insights about their performance across different text types (only if sufficient data exists)
        3. Pattern-based recommendations for continued reading development (limited to reliable observations)
        4. 3-4 evidence-based teaching strategies that would address their specific needs
        
        Format your response with clearly defined sections and explicitly mention data quality issues where relevant.
        """

        result = await agent.run(prompt)

        if not result or not result.data:
            raise ValueError("Failed to generate cumulative analysis")

        return result.data

    except Exception as e:
        print(f"AI Cumulative Analysis Generation Error: {str(e)}")
        # Provide fallback response if AI fails
        return CumulativeAnalysis(
            progression_analysis="Unable to generate a detailed progression analysis due to data quality issues. The available data shows inconsistent patterns that require additional assessments for reliable conclusions.",
            text_performance_insights="The current data shows varying performance across text types, but more consistent assessment data is needed for reliable conclusions. Note that category scores only reflect questions the student attempted, which may lead to misleading conclusions when few attempts were made.",
            development_recommendations="Based on the limited reliable data available, continue to provide varied reading materials and monitor progress with additional assessments to establish more reliable patterns.",
            teaching_strategies=[
                "Use guided reading with texts at appropriate levels",
                "Implement comprehension strategy instruction across multiple text types",
                "Provide regular feedback on specific reading skills",
                "Conduct additional assessments to establish more reliable performance patterns",
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


@router.get("/student/{student_id}/cumulative")
async def get_student_cumulative_report(
    student_id: str,
    db: Session = Depends(get_db),
    teacher: User = Depends(require_teacher),
    regenerate: bool = False,
):
    """Generate a cumulative report with analysis across multiple assessments for a student"""

    if not CUMULATIVE_REPORTS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cumulative reports are temporarily disabled",
        )

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
                status_code=status.HTTP_404_NOT_FOUND, detail="Student not found"
            )

        # Get all completed assessments for this student where the texts were created by this teacher
        completions = (
            db.query(Completion)
            .join(Text, Completion.text_id == Text.id)
            .filter(
                Completion.student_id == student_id,
                Completion.is_deleted == False,
                Completion.test_status == "completed",
                Text.teacher_id == teacher.id,
                Text.is_deleted == False,
            )
            .order_by(Completion.completed_at)
            .all()
        )

        # Check if there are enough completions for a cumulative report
        if len(completions) < MIN_NUMBER_COMPLETIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"At least {MIN_NUMBER_COMPLETIONS} completed assessments are required for a cumulative report",
            )

        # Get all related texts
        text_ids = [c.text_id for c in completions]
        texts = db.query(Text).filter(Text.id.in_(text_ids)).all()
        text_map = {text.id: text for text in texts}

        # Calculate improved category stats with proper data quality indicators
        improved_stats = generate_improved_category_stats(completions)
        category_stats = improved_stats["stats"]
        data_quality_indicators = improved_stats["data_quality"]

        # Check if analysis exists in cache for this student and we're not forcing regeneration
        # We'll use the most recent completion's id as part of the cache key
        latest_completion = completions[-1]
        cache_key = f"cumulative_{student_id}_{latest_completion.id}"

        # Look for cached analysis in the database
        if not regenerate and latest_completion.analysis_content:
            try:
                # Try to parse the cached analysis
                cached_data = json.loads(latest_completion.analysis_content)
                if "cumulative_analysis" in cached_data:
                    analysis = CumulativeAnalysis.parse_obj(
                        cached_data["cumulative_analysis"]
                    )
                    from_cache = True
                else:
                    # Generate new analysis if the cached data doesn't have cumulative analysis
                    analysis = await generate_cumulative_analysis(completions, texts)

                    # Store the analysis in the latest completion record
                    cache_data = {"cumulative_analysis": analysis.dict()}
                    latest_completion.analysis_content = json.dumps(cache_data)
                    db.commit()
                    from_cache = False
            except Exception as e:
                # If parsing fails, generate new analysis
                print(f"Error parsing cached analysis: {str(e)}")
                analysis = await generate_cumulative_analysis(completions, texts)

                # Store the analysis in the latest completion record
                cache_data = {"cumulative_analysis": analysis.dict()}
                latest_completion.analysis_content = json.dumps(cache_data)
                db.commit()
                from_cache = False
        else:
            # Generate new analysis
            analysis = await generate_cumulative_analysis(completions, texts)

            # Store the analysis in the latest completion record
            cache_data = {"cumulative_analysis": analysis.dict()}
            latest_completion.analysis_content = json.dumps(cache_data)
            db.commit()
            from_cache = False

        # Format category performance data in a way that clearly indicates data quality
        category_performance = []
        for category, stats in category_stats.items():
            quality = data_quality_indicators.get(category, {})
            reliability = (
                "low"
                if quality.get("limited_data", True)
                else "medium" if quality.get("potential_misleading", False) else "high"
            )

            category_performance.append(
                {
                    "category": category,
                    "score": stats["score"],
                    "attempts": stats["total_questions"],
                    "assessments_with_data": stats["assessments_with_data"],
                    "reliability": reliability,
                    "data_quality_issues": [
                        issue
                        for issue in ["limited_data", "inconsistent_coverage"]
                        if quality.get(
                            (
                                "limited_data"
                                if issue == "limited_data"
                                else "potential_misleading"
                            ),
                            False,
                        )
                    ],
                }
            )

        # Calculate text type performance with reliability indicators
        text_performance = {}
        text_type_attempts = {}
        for completion in completions:
            text = text_map.get(completion.text_id)
            if not text:
                continue

            if text.type_name not in text_performance:
                text_performance[text.type_name] = []
                text_type_attempts[text.type_name] = 0

            text_performance[text.type_name].append(completion.overall_score)
            text_type_attempts[text.type_name] += 1

        # Convert to averages and add reliability indicators
        text_performance_with_reliability = {}
        for type_name, scores in text_performance.items():
            attempts = text_type_attempts[type_name]
            reliability = (
                "high" if attempts >= 3 else "medium" if attempts == 2 else "low"
            )

            text_performance_with_reliability[type_name] = {
                "score": sum(scores) / len(scores),
                "attempts": attempts,
                "reliability": reliability,
            }

        # Calculate date range
        first_date = min(c.completed_at for c in completions)
        last_date = max(c.completed_at for c in completions)
        days_covered = (
            last_date - first_date
        ).days + 1  # Include both first and last day

        # Calculate score range
        scores = [c.overall_score for c in completions]
        avg_score = sum(scores) / len(scores)
        lowest_score = min(scores)
        highest_score = max(scores)

        # Compile completion data for time series
        completion_history = []
        for completion in completions:
            text = text_map.get(completion.text_id)
            if not text:
                continue

            completion_history.append(
                {
                    "date": completion.completed_at,
                    "score": completion.overall_score,
                    "level": completion.final_test_level,
                    "textId": completion.text_id,
                    "textTitle": text.title,
                    "textType": text.type_name,
                    "gradeLevel": text.grade_level,
                }
            )

        # Determine if there's sufficient data for a meaningful analysis
        has_sufficient_data = any(
            cat.get("reliability", "low") != "low" for cat in category_performance
        )

        # Determine overall data quality
        overall_data_quality = "high"
        if not has_sufficient_data:
            overall_data_quality = "low"
        elif any(
            cat.get("reliability", "") == "medium" for cat in category_performance
        ):
            overall_data_quality = "medium"

        # Prepare an overall data quality report
        data_quality_report = {
            "overall_quality": overall_data_quality,
            "has_sufficient_data": has_sufficient_data,
            "category_coverage": sum(
                1 for cat in category_performance if cat.get("attempts", 0) > 0
            )
            / 7,  # 7 is the total number of categories
            "assessment_count": len(completions),
            "category_indicators": {
                cat["category"]: cat["reliability"] for cat in category_performance
            },
            "has_data_discrepancies": any(
                quality.get("potential_misleading", False)
                for quality in data_quality_indicators.values()
            ),
            "note": "Category scores reflect only questions the student attempted in each category. Categories with no attempts do not appear in the report. Text type scores reflect overall performance on all questions in that text type.",
        }

        return {
            "student_name": user.full_name,
            "report_type": "cumulative",
            "analysis": analysis.dict(),
            "grade_level": (
                texts[0].grade_level if texts else None
            ),  # Using first text's grade level
            "tests_count": len(completions),
            "first_assessment_date": first_date,
            "latest_assessment_date": last_date,
            "days_covered": days_covered,
            "average_score": avg_score,
            "lowest_score": lowest_score,
            "highest_score": highest_score,
            "category_performance": category_performance,
            "text_performance": text_performance_with_reliability,
            "completion_history": completion_history,
            "from_cache": from_cache,
            "data_quality": data_quality_report,
        }

    except Exception as e:
        import traceback

        print(f"DEBUG: Exception caught: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating cumulative report: {str(e)}",
        )

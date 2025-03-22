# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import os

from database.session import get_db, engine
from database.models import (
    Base,
    User,
    Role,
    QuestionCategory,
    QuestionDifficulty,
    EmailWhitelist,
    WhitelistType,
)
from verification.session import engine as verification_engine
from verification.base import Base as VerificationBase

from routers.auth.login import router as login_router
from routers.auth.registration import router as registration_router
from routers.auth.token import router as token_router
from routers.admin.users import router as user_router
from routers.admin.whitelist import router as whitelist_router  # New import
from routers.teacher.texts import router as text_router
from routers.student.teachers import router as student_teachers_router
from routers.auth.logout import router as logout_router
from routers.student.assessment import router as assessment_router
from routers.student.questions import router as questions_router
from routers.student.evaluation import router as evaluation_router
from routers.student.completion import router as completion_router
from routers.student.completion_test import router as completion_test_router
from routers.teacher.reports import router as teacher_reports_router
from routers.student.simplify import router as simplify_router
from routers.admin.cache import router as cache_admin_router
from routers.admin.question_cache import router as question_cache_router
from services.whitelist_service import WhitelistService
from routers.admin.database import router as database_admin_router
from routers.teacher.bypass import router as teacher_bypass_router


load_dotenv()


async def create_db_and_tables():
    """Create database tables if they don't exist"""
    Base.metadata.create_all(bind=engine)
    VerificationBase.metadata.create_all(bind=verification_engine)


async def setup_initial_data():
    """Initialize reference data"""
    db = next(get_db())
    try:
        # Create roles if they don't exist
        admin_role = db.query(Role).filter(Role.role_name == "ADMIN").first()
        if not admin_role:
            roles = [
                Role(role_name="ADMIN", description="System administrator"),
                Role(role_name="TEACHER", description="Can create and manage texts"),
                Role(role_name="STUDENT", description="Can take assessments"),
            ]
            db.add_all(roles)
            db.commit()

        # Create question categories if they don't exist
        categories = db.query(QuestionCategory).count()
        if categories == 0:
            print("Initializing question categories...")
            question_categories = [
                QuestionCategory(
                    category_name="literal_basic",
                    description="Basic understanding of explicitly stated information",
                    progression_order=1,
                ),
                QuestionCategory(
                    category_name="literal_detailed",
                    description="Detailed understanding of explicitly stated information",
                    progression_order=2,
                ),
                QuestionCategory(
                    category_name="vocabulary_context",
                    description="Understanding vocabulary in context",
                    progression_order=3,
                ),
                QuestionCategory(
                    category_name="inferential_simple",
                    description="Simple inferences from text",
                    progression_order=4,
                ),
                QuestionCategory(
                    category_name="inferential_complex",
                    description="Complex inferences requiring deeper analysis",
                    progression_order=5,
                ),
                QuestionCategory(
                    category_name="structural_basic",
                    description="Basic understanding of text structure",
                    progression_order=6,
                ),
                QuestionCategory(
                    category_name="structural_advanced",
                    description="Advanced analysis of text structure and organization",
                    progression_order=7,
                ),
            ]
            db.add_all(question_categories)
            db.commit()
            print("Question categories initialized successfully.")

        # Create question difficulty levels if they don't exist
        difficulties = db.query(QuestionDifficulty).count()
        if difficulties == 0:
            print("Initializing question difficulty levels...")
            question_difficulties = [
                QuestionDifficulty(
                    difficulty_name="basic",
                    description="Entry level questions",
                    level_value=1,
                ),
                QuestionDifficulty(
                    difficulty_name="intermediate",
                    description="Medium difficulty questions",
                    level_value=2,
                ),
                QuestionDifficulty(
                    difficulty_name="advanced",
                    description="Challenging questions",
                    level_value=3,
                ),
            ]
            db.add_all(question_difficulties)
            db.commit()
            print("Question difficulty levels initialized successfully.")

        # Setup initial whitelist if needed
        whitelist_count = db.query(EmailWhitelist).count()
        if whitelist_count == 0:
            print("Setting up initial email whitelist...")

            # Get the admin email domain for initial whitelist
            initial_admin_email = os.getenv("INITIAL_ADMIN_EMAIL")
            if initial_admin_email and "@" in initial_admin_email:
                admin_domain = initial_admin_email.split("@")[-1]

                # Add the admin domain to whitelist
                WhitelistService.add_to_whitelist(
                    admin_domain, WhitelistType.DOMAIN, "Initial admin domain", db
                )

                # Also whitelist the specific admin email
                WhitelistService.add_to_whitelist(
                    initial_admin_email, WhitelistType.EMAIL, "Initial admin email", db
                )

                print(f"Added initial admin domain '{admin_domain}' to whitelist")

            # Add sample school district domain if specified in env
            school_domain = os.getenv("SCHOOL_DOMAIN")
            if school_domain:
                WhitelistService.add_to_whitelist(
                    school_domain, WhitelistType.DOMAIN, "School district domain", db
                )
                print(f"Added school domain '{school_domain}' to whitelist")

            print("Email whitelist initialized successfully.")

    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database setup"""
    # Create tables and initial data
    await create_db_and_tables()
    await setup_initial_data()
    yield
    # Cleanup can go here if needed


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, debug=True)

app.include_router(login_router, prefix="/auth")
app.include_router(registration_router, prefix="/auth")
app.include_router(token_router, prefix="/auth")
app.include_router(user_router, prefix="/admin")
app.include_router(whitelist_router, prefix="/admin")  # New router
app.include_router(text_router, prefix="/teacher")
app.include_router(student_teachers_router, prefix="/student/teachers")
app.include_router(logout_router, prefix="/auth")
app.include_router(assessment_router, prefix="/assessment")
app.include_router(questions_router, prefix="/questions")
app.include_router(evaluation_router, prefix="/evaluation")
app.include_router(completion_router, prefix="/student/completion")
app.include_router(completion_test_router, prefix="/completion-test")
app.include_router(teacher_reports_router, prefix="/teacher/reports")
app.include_router(simplify_router, prefix="/simplify")
app.include_router(cache_admin_router, prefix="/admin/cache")
app.include_router(question_cache_router, prefix="/admin/question-cache")
app.include_router(database_admin_router, prefix="/admin")
app.include_router(teacher_bypass_router, prefix="/teacher")


from routers.admin.cache import (
    router as cache_admin_router,
)  # New import for cache management


origins = [
    os.getenv("FRONTEND_URL", "http://localhost:5173"),
    "https://umaread.org",
    "http://umaread.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Example endpoint using the database
@app.get("/users/{user_id}")
def read_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    return user

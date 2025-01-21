import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.base import Base
from database.models import (
    Role,
    TextForm,
    PrimaryType,
    QuestionCategory,
    QuestionDifficulty,
    Genre,
)


@pytest.fixture(scope="session")
def engine():
    """Create a test database engine"""
    return create_engine("sqlite:///:memory:")


@pytest.fixture(scope="session")
def tables(engine):
    """Create all tables for testing"""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(engine, tables):
    """Create a new database session for a test"""
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    # Insert reference data
    roles = [
        Role(role_name="STUDENT", description="Student role"),
        Role(role_name="TEACHER", description="Teacher role"),
        Role(role_name="ADMIN", description="Admin role"),
    ]

    text_forms = [
        TextForm(form_name="PROSE", description="Standard written text"),
        TextForm(form_name="POETRY", description="Verse format"),
        TextForm(form_name="DRAMA", description="Script/dialogue format"),
        TextForm(form_name="OTHER", description="Alternative text formats"),
    ]

    primary_types = [
        PrimaryType(type_name="NARRATIVE", description="Story-based text"),
        PrimaryType(type_name="INFORMATIONAL", description="Factual content"),
        PrimaryType(type_name="PERSUASIVE", description="Argumentative text"),
        PrimaryType(type_name="OTHER", description="Alternative text types"),
    ]

    question_categories = [
        QuestionCategory(
            category_name="literal_basic",
            description="Basic fact recall",
            progression_order=1,
        ),
        QuestionCategory(
            category_name="literal_detailed",
            description="Detailed fact recall",
            progression_order=2,
        ),
        QuestionCategory(
            category_name="vocabulary_context",
            description="Word meaning in context",
            progression_order=3,
        ),
        QuestionCategory(
            category_name="inferential_simple",
            description="Basic conclusions",
            progression_order=4,
        ),
        QuestionCategory(
            category_name="inferential_complex",
            description="Complex conclusions",
            progression_order=5,
        ),
        QuestionCategory(
            category_name="structural_basic",
            description="Basic text structure",
            progression_order=6,
        ),
        QuestionCategory(
            category_name="structural_advanced",
            description="Advanced text structure",
            progression_order=7,
        ),
    ]

    question_difficulties = [
        QuestionDifficulty(
            difficulty_name="basic", description="Entry level questions", level_value=1
        ),
        QuestionDifficulty(
            difficulty_name="intermediate",
            description="Medium complexity",
            level_value=2,
        ),
        QuestionDifficulty(
            difficulty_name="advanced", description="High complexity", level_value=3
        ),
    ]

    genres = [
        Genre(genre_name="FANTASY", description="Imaginative fiction"),
        Genre(genre_name="MYTHOLOGY", description="Traditional stories"),
        Genre(genre_name="REALISTIC", description="True-to-life fiction"),
        Genre(genre_name="HISTORICAL", description="Based on history"),
        Genre(genre_name="TECHNICAL", description="Specialized/technical content"),
        Genre(genre_name="BIOGRAPHY", description="Life stories"),
        Genre(genre_name="ADVENTURE", description="Action-based stories"),
        Genre(genre_name="MYSTERY", description="Problem-solving stories"),
        Genre(genre_name="NONFICTION", description="Factual content"),
        Genre(genre_name="OTHER", description="Miscellaneous genres"),
    ]

    session.add_all(roles)
    session.add_all(text_forms)
    session.add_all(primary_types)
    session.add_all(question_categories)
    session.add_all(question_difficulties)
    session.add_all(genres)
    session.commit()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user"""
    from database.models import User

    user = User(
        email="test@example.com",
        username="teststudent",
        full_name="Test Student",
        role_name="STUDENT",
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def test_teacher(db_session):
    """Create a test teacher"""
    from database.models import User

    teacher = User(
        email="teacher@example.com",
        username="testteacher",
        full_name="Test Teacher",
        role_name="TEACHER",
    )
    db_session.add(teacher)
    db_session.commit()
    return teacher

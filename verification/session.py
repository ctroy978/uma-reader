from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from .base import Base

VERIFICATION_DB_URL = os.getenv("VERIFICATION_DB_URL", "sqlite:///./verification.db")

engine = create_engine(
    VERIFICATION_DB_URL, connect_args={"check_same_thread": False}  # for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_verification_tables():
    """Create all tables in the verification database"""
    Base.metadata.create_all(bind=engine)


def get_verification_db():
    """Dependency for getting verification database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

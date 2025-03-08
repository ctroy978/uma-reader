import os
import logging
from typing import Generator  # Add this import
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from .base import Base  # Assuming Base is in verification/base.py

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Get database URL from environment
VERIFICATION_DB_URL = os.getenv("VERIFICATION_DB_URL", "sqlite:///./verification.db")

# Extract the database file path for logging
VERIFICATION_DB_FILE = VERIFICATION_DB_URL.replace("sqlite:///", "")

# Log database status
if os.path.exists(VERIFICATION_DB_FILE) and os.path.getsize(VERIFICATION_DB_FILE) > 0:
    logger.info(f"Existing verification database detected at {VERIFICATION_DB_FILE}")
else:
    logger.info(
        f"No existing verification database at {VERIFICATION_DB_FILE}; a new one will be created."
    )

# Create SQLAlchemy engine with plain SQLite
engine = create_engine(
    VERIFICATION_DB_URL,
    connect_args={"check_same_thread": False},
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_verification_tables():
    """Create all tables in the verification database"""
    Base.metadata.create_all(bind=engine)


def get_verification_db() -> Generator[Session, None, None]:  # Updated return type
    """Dependency for getting verification database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

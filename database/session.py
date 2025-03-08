import os
import sys
import logging
import sqlite3
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import sqlcipher3

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (useful for development)
load_dotenv()

# Get database URL and encryption key from environment, with defaults for development
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./reading_assessment.db")
ENCRYPTION_KEY = os.getenv("DATABASE_ENCRYPTION_KEY", "default_key_for_development")

# Extract the database file path from the URL
DB_FILE = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")

# Optional: Fail-fast if an existing unencrypted DB is detected (set via env var)
FAIL_ON_EXISTING_UNENCRYPTED = (
    os.getenv("FAIL_ON_EXISTING_UNENCRYPTED", "false").lower() == "true"
)

# Check if the database file already exists and verify encryption
if os.path.exists(DB_FILE) and os.path.getsize(DB_FILE) > 0:
    try:
        # Attempt to open without the key; if it succeeds, the DB isn't encrypted
        conn = sqlite3.connect(DB_FILE)
        conn.close()
        msg = f"Existing database file at {DB_FILE} is not encrypted!"
        if FAIL_ON_EXISTING_UNENCRYPTED:
            logger.error(msg)
            sys.exit(1)
        else:
            logger.warning(msg)
    except sqlite3.DatabaseError:
        # Exception means it’s encrypted (or corrupted, but we assume encryption)
        logger.info(f"Existing encrypted database detected at {DB_FILE}")
else:
    logger.info(f"No existing database at {DB_FILE}; a new one will be created.")

# Create SQLAlchemy engine with SQLCipher
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    module=sqlcipher3.dbapi2,
    connect_args={"check_same_thread": False},
)


# Set up SQLCipher encryption PRAGMA settings
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute(f"PRAGMA key = '{ENCRYPTION_KEY}'")
    cursor.execute("PRAGMA cipher_page_size = 4096")
    cursor.execute("PRAGMA kdf_iter = 64000")
    cursor.execute("PRAGMA cipher_hmac_algorithm = HMAC_SHA512")
    cursor.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512")
    cursor.close()


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency for getting database sessions
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

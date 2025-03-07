# database/session.py
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
import os
from typing import Generator
from dotenv import load_dotenv
import sqlcipher3  # Add this explicit import

load_dotenv()
ENCRYPTION_KEY = os.getenv("DB_ENCRYPTION_KEY", "default_key_for_development")

# Make sure database file doesn't exist yet (for first run)
DB_FILE = "./reading_assessment.db"
if os.path.exists(DB_FILE) and os.path.getsize(DB_FILE) > 0:
    print(
        "NOTE: Using existing database file. If this wasn't encrypted from the start, it won't be encrypted now."
    )

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"

# Use connect_args to explicitly specify sqlcipher
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    module=sqlcipher3.dbapi2,  # Force use of SQLCipher
    connect_args={"check_same_thread": False},
)


# Encryption setup
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute(f"PRAGMA key = '{ENCRYPTION_KEY}'")
    cursor.execute("PRAGMA cipher_page_size = 4096")
    cursor.execute("PRAGMA kdf_iter = 64000")
    cursor.execute("PRAGMA cipher_hmac_algorithm = HMAC_SHA512")
    cursor.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

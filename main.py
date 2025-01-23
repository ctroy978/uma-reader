# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import os

from database.session import get_db, engine
from database.models import Base, User, Role
from verification.session import engine as verification_engine
from verification.base import Base as VerificationBase

from routers import auth

load_dotenv()


async def create_db_and_tables():
    """Create database tables if they don't exist"""
    Base.metadata.create_all(bind=engine)
    VerificationBase.metadata.create_all(bind=verification_engine)


async def setup_initial_data():
    """Initialize reference data and admin user if needed"""
    db = next(get_db())
    try:
        # Check if admin role exists
        admin_role = db.query(Role).filter(Role.role_name == "ADMIN").first()
        if not admin_role:
            # Create roles
            roles = [
                Role(role_name="ADMIN", description="System administrator"),
                Role(role_name="TEACHER", description="Can create and manage texts"),
                Role(role_name="STUDENT", description="Can take assessments"),
            ]
            db.add_all(roles)
            db.commit()  # Commit roles first

        # Check if admin user exists
        admin_user = db.query(User).join(Role).filter(Role.role_name == "ADMIN").first()
        if not admin_user:
            admin_user = User(
                email="admin@example.com",
                username="admin",
                full_name="System Administrator",
                role_name="ADMIN",
            )
            db.add(admin_user)

        db.commit()
    except Exception as e:
        db.rollback()
        raise e
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
app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)


origins = [
    os.getenv("FRONTEND_URL", "http://localhost:5173"),
    # Add more origins as needed
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

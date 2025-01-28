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

from routers.auth.login import router as login_router
from routers.auth.registration import router as registration_router
from routers.auth.token import router as token_router


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

app.include_router(login_router, prefix="/auth")
app.include_router(registration_router, prefix="/auth")
app.include_router(token_router, prefix="/auth")


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

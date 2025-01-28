import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

from routers.auth.token import router
from auth.jwt_utils import create_access_token, create_refresh_token
from database.models.token import RefreshToken
from database.session import get_db
from verification.session import get_verification_db

# Create test app
app = FastAPI()
app.include_router(router)


# Override dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_verification_db] = override_get_db

# Create test client - fixed initialization
client = TestClient(app=app.router)


class MockRequest:
    def __init__(self, headers=None, client=None):
        self.headers = headers or {}
        self.client = client or type("Client", (), {"host": "127.0.0.1"})


@pytest.fixture
def access_token(test_user):
    return create_access_token(test_user)


@pytest.fixture
def refresh_token(test_user, db_session):
    request = MockRequest()
    return create_refresh_token(test_user, db_session, request)


def test_refresh_token_success(db_session_with_verification, test_user, refresh_token):
    """Test successful token refresh"""
    client.cookies.set("refresh_token", refresh_token)
    response = client.post("/auth/token/refresh")

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data
    assert data["role"] == test_user.role_name
    assert "refresh_token" in response.cookies


def test_refresh_token_no_cookie(db_session_with_verification):
    """Test refresh attempt without token cookie"""
    response = client.post("/auth/token/refresh")
    assert response.status_code == 401
    assert response.json()["detail"] == "No refresh token provided"


def test_refresh_token_invalid(db_session_with_verification):
    """Test refresh with invalid token"""
    client.cookies.set("refresh_token", "invalid.token.here")
    response = client.post("/auth/token/refresh")
    assert response.status_code == 401


def test_revoke_token(db_session_with_verification, access_token, refresh_token):
    """Test token revocation"""
    client.cookies.set("refresh_token", refresh_token)
    headers = {"Authorization": f"Bearer {access_token}"}

    response = client.post("/auth/token/revoke", headers=headers)
    assert response.status_code == 200
    assert "refresh_token" not in response.cookies

    # Verify tokens are blacklisted
    response = client.post("/auth/token/refresh")
    assert response.status_code == 401


def test_verify_token_endpoints(db_session_with_verification, test_user):
    """Test all token verification endpoints"""
    # Test /auth/token/verify
    token = create_access_token(test_user)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post("/auth/token/verify", headers=headers)
    assert response.status_code == 200
    assert response.json()["valid"] == True

    # Test with invalid token
    headers = {"Authorization": "Bearer invalid.token"}
    response = client.post("/auth/token/verify", headers=headers)
    assert response.status_code == 401


@pytest.fixture(autouse=True)
def setup_db(db_session_with_verification):
    """Setup test database for each test"""
    global TestingSessionLocal
    TestingSessionLocal = lambda: db_session_with_verification
    yield
    db_session_with_verification.rollback()

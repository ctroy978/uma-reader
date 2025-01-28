import pytest
from fastapi import HTTPException
from auth.middleware import AuthMiddleware, get_optional_user
from auth.jwt_utils import create_access_token

import pytest

pytestmark = pytest.mark.asyncio


class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.state = type("State", (), {})()


@pytest.fixture
def auth_middleware():
    return AuthMiddleware()


@pytest.fixture
def role_middleware():
    return AuthMiddleware(allowed_roles=["TEACHER", "ADMIN"])


async def test_auth_middleware_no_credentials(auth_middleware):
    """Test middleware rejects requests without credentials"""
    with pytest.raises(HTTPException) as exc:
        await auth_middleware(MockRequest(), None, None, None)
    assert exc.value.status_code == 401


async def test_auth_middleware_valid_token(
    auth_middleware, test_user, db_session_with_verification
):
    """Test middleware accepts valid token"""
    token = create_access_token(test_user)
    credentials = type("Credentials", (), {"credentials": token})()
    request = MockRequest()

    user = await auth_middleware(
        request, credentials, db_session_with_verification, db_session_with_verification
    )

    assert user.id == test_user.id
    assert hasattr(request.state, "user")
    assert request.state.user.id == test_user.id


async def test_role_middleware_unauthorized(
    role_middleware, test_user, db_session_with_verification
):
    """Test middleware rejects user with invalid role"""
    token = create_access_token(test_user)  # test_user is STUDENT
    credentials = type("Credentials", (), {"credentials": token})()

    with pytest.raises(HTTPException) as exc:
        await role_middleware(
            MockRequest(),
            credentials,
            db_session_with_verification,
            db_session_with_verification,
        )
    assert exc.value.status_code == 403


async def test_get_optional_user_no_token(db_session_with_verification):
    """Test optional auth returns None without token"""
    request = MockRequest()
    user = await get_optional_user(
        request, db_session_with_verification, db_session_with_verification
    )
    assert user is None


async def test_get_optional_user_valid_token(test_user, db_session_with_verification):
    """Test optional auth returns user with valid token"""
    token = create_access_token(test_user)
    request = MockRequest(headers={"Authorization": f"Bearer {token}"})

    user = await get_optional_user(
        request, db_session_with_verification, db_session_with_verification
    )

    assert user is not None
    assert user.id == test_user.id

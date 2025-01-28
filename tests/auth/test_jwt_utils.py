import pytest
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from auth.jwt_utils import (
    create_token_payload,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token,
    blacklist_token,
    get_token_from_header,
)
from auth.jwt_config import get_jwt_settings, get_token_settings
from verification.models import TokenBlacklist
from database.models import RefreshToken
from verification.base import Base as VerificationBase

settings = get_jwt_settings()
token_settings = get_token_settings()


class MockRequest:
    def __init__(self, headers=None, client=None):
        self.headers = headers or {}
        self.client = client or type("Client", (), {"host": "127.0.0.1"})


@pytest.fixture(scope="function")
def db_session_with_verification(db_session, engine):
    """Create verification tables in test database"""
    VerificationBase.metadata.create_all(engine)
    yield db_session
    VerificationBase.metadata.drop_all(engine)


def test_create_token_payload(test_user):
    """Test token payload creation"""
    payload = create_token_payload(test_user, "access")

    assert isinstance(payload, dict)
    assert payload["sub"] == str(test_user.id)
    assert payload["role"] == test_user.role_name
    assert payload["type"] == "access"
    assert "jti" in payload
    assert "iat" in payload
    assert "exp" in payload

    # Verify expiration
    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
    assert exp > iat


def test_create_access_token(test_user):
    """Test access token creation"""
    token = create_access_token(test_user)

    assert isinstance(token, str)
    payload = decode_token(token)
    assert payload["type"] == "access"
    assert payload["sub"] == str(test_user.id)


def test_create_refresh_token(test_user, db_session):
    """Test refresh token creation and storage"""
    request = MockRequest()
    token = create_refresh_token(test_user, db_session, request)

    assert isinstance(token, str)
    payload = decode_token(token)
    assert payload["type"] == "refresh"

    # Verify storage
    stored_token = db_session.query(RefreshToken).filter_by(token=token).first()
    assert stored_token is not None
    assert stored_token.user_id == test_user.id
    assert stored_token.issued_by_ip == "127.0.0.1"


def test_decode_token(test_user):
    """Test token decoding"""
    token = create_access_token(test_user)
    payload = decode_token(token)

    assert isinstance(payload, dict)
    assert payload["sub"] == str(test_user.id)
    assert payload["role"] == test_user.role_name


def test_decode_invalid_token():
    """Test invalid token decoding"""
    with pytest.raises(HTTPException) as exc:
        decode_token("invalid.token.here")
    assert exc.value.status_code == 401


def test_verify_token(test_user, db_session_with_verification):
    """Test token verification"""
    token = create_access_token(test_user)
    payload = verify_token(token, db_session_with_verification, "access")

    assert payload["sub"] == str(test_user.id)
    assert payload["type"] == "access"


def test_verify_token_wrong_type(test_user, db_session_with_verification):
    token = create_access_token(test_user)
    with pytest.raises(HTTPException) as exc:
        verify_token(token, db_session_with_verification, "refresh")
    assert exc.value.status_code == 401


def test_blacklist_token(test_user, db_session_with_verification):
    token = create_access_token(test_user)
    blacklist_token(token, db_session_with_verification)
    payload = decode_token(token)
    assert TokenBlacklist.is_blacklisted(db_session_with_verification, payload["jti"])


def test_get_token_from_header():
    """Test token extraction from header"""
    token = "test.token.here"
    request = MockRequest(headers={"Authorization": f"Bearer {token}"})

    extracted = get_token_from_header(request)
    assert extracted == token


def test_get_token_from_header_invalid():
    """Test invalid header token extraction"""
    cases = [
        {},  # No header
        {"Authorization": ""},  # Empty header
        {"Authorization": "Bearer"},  # No token
        {"Authorization": "NotBearer token"},  # Wrong scheme
    ]

    for headers in cases:
        request = MockRequest(headers=headers)
        assert get_token_from_header(request) is None

# app/services/whitelist_service.py
from sqlalchemy.orm import Session
import os

from database.models import EmailWhitelist, WhitelistType


class WhitelistService:
    """Service for checking email against whitelist"""

    @staticmethod
    def is_email_allowed(email: str, db: Session) -> bool:
        """
        Check if an email is allowed to register

        Conditions for allowed registration:
        1. Email matches the INITIAL_ADMIN_EMAIL environment variable
        2. Email domain is in the whitelist domains
        3. Exact email is in the whitelist emails
        """
        if not email:
            return False

        # Always allow the initial admin email
        initial_admin_email = os.getenv("INITIAL_ADMIN_EMAIL")
        if initial_admin_email and email.lower() == initial_admin_email.lower():
            return True

        # Normalize email
        email = email.lower()
        domain = email.split("@")[-1] if "@" in email else None

        if not domain:
            return False

        # Check for exact email match
        email_match = (
            db.query(EmailWhitelist)
            .filter(
                EmailWhitelist.type == WhitelistType.EMAIL,
                EmailWhitelist.value == email,
                EmailWhitelist.is_deleted == False,
            )
            .first()
        )

        if email_match:
            return True

        # Check for domain match
        domain_match = (
            db.query(EmailWhitelist)
            .filter(
                EmailWhitelist.type == WhitelistType.DOMAIN,
                EmailWhitelist.value == domain,
                EmailWhitelist.is_deleted == False,
            )
            .first()
        )

        return domain_match is not None

    @staticmethod
    def add_to_whitelist(
        value: str, whitelist_type: WhitelistType, description: str, db: Session
    ) -> EmailWhitelist:
        """Add a new entry to the whitelist"""
        existing = (
            db.query(EmailWhitelist)
            .filter(EmailWhitelist.value == value.lower())
            .first()
        )

        if existing:
            if existing.is_deleted:
                # Reactivate soft-deleted entry
                existing.is_deleted = False
                existing.type = whitelist_type
                existing.description = description
                db.commit()
                return existing
            else:
                # Entry already exists and is active
                return existing

        # Create new entry
        whitelist_entry = EmailWhitelist(
            value=value.lower(), type=whitelist_type, description=description
        )

        db.add(whitelist_entry)
        db.commit()
        db.refresh(whitelist_entry)

        return whitelist_entry

    @staticmethod
    def remove_from_whitelist(id: str, db: Session) -> bool:
        """Soft delete a whitelist entry"""
        entry = db.query(EmailWhitelist).filter(EmailWhitelist.id == id).first()

        if not entry:
            return False

        entry.is_deleted = True
        db.commit()

        return True

    @staticmethod
    def get_all_whitelist_entries(db: Session):
        """Get all active whitelist entries"""
        return (
            db.query(EmailWhitelist)
            .filter(EmailWhitelist.is_deleted == False)
            .order_by(EmailWhitelist.type, EmailWhitelist.value)
            .all()
        )

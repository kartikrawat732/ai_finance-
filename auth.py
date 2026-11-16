"""Email and phone-number authentication service with PBKDF2 password hashing and OTP."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import html
import logging
import os
import re
import secrets
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional, Union

from database import (
    consume_otp_challenge,
    create_user,
    get_filter_metadata,
    get_user_by_identifier,
    seed_sample_data,
    store_otp_challenge,
)

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_PHONE_CLEAN_RE = re.compile(r"[\s().-]")
_PHONE_VALID_RE = re.compile(r"^\+?[1-9]\d{7,14}$")
_PBKDF2_ITERATIONS = 200_000


def normalize_identifier(raw: str) -> tuple[str, str]:
    """Validate and normalize an email address or phone number.

    Returns ``(normalized_value, kind)`` where ``kind`` is ``"email"`` or ``"phone"``.
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("Please enter an email address or phone number.")

    if "@" in value:
        email = value.lower()
        if not _EMAIL_RE.match(email):
            raise ValueError("Please enter a valid email address (e.g., alex@example.com).")
        return email, "email"

    cleaned_phone = _PHONE_CLEAN_RE.sub("", value)
    if cleaned_phone.startswith("00"):
        cleaned_phone = f"+{cleaned_phone[2:]}"
    if not cleaned_phone.startswith("+") and cleaned_phone.isdigit() and len(cleaned_phone) == 10:
        # Preserve canonical 10-digit normalization without forcing a country code
        return cleaned_phone, "phone"
    if not _PHONE_VALID_RE.match(cleaned_phone):
        raise ValueError(
            "Please enter a valid email address or 8-15 digit phone number (e.g., +919876543210 or 9876543210)."
        )
    return cleaned_phone, "phone"


def hash_password(password: str) -> str:
    """Hash a password using salted PBKDF2-HMAC-SHA256."""
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        _PBKDF2_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash in constant time."""
    if not password or not stored_hash:
        return False
    parts = stored_hash.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        iterations = int(parts[1])
        salt_bytes = bytes.fromhex(parts[2])
        expected_hex = parts[3]
    except ValueError:
        return False

    candidate_hex = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        iterations,
    ).hex()
    return hmac.compare_digest(candidate_hex, expected_hex)


def _public_user_dict(user_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(user_row["id"]),
        "full_name": str(user_row["full_name"]),
        "email": user_row.get("email"),
        "phone": user_row.get("phone"),
        "created_at": user_row.get("created_at"),
    }


def register_user(
    full_name: str,
    identifier: str,
    password: str,
    *,
    secondary_identifier: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Register a new user using email and/or phone number authentication."""
    clean_name = (full_name or "").strip()
    if len(clean_name) < 2:
        raise ValueError("Full name must be at least 2 characters.")

    primary_val, primary_kind = normalize_identifier(identifier)
    email: Optional[str] = primary_val if primary_kind == "email" else None
    phone: Optional[str] = primary_val if primary_kind == "phone" else None

    if secondary_identifier and secondary_identifier.strip():
        sec_val, sec_kind = normalize_identifier(secondary_identifier)
        if sec_kind == primary_kind:
            raise ValueError(
                f"Secondary contact must be a {'phone number' if primary_kind == 'email' else 'email address'}."
            )
        if sec_kind == "email":
            email = sec_val
        else:
            phone = sec_val

    pw_hash = hash_password(password)
    created = create_user(
        full_name=clean_name,
        email=email,
        phone=phone,
        password_hash=pw_hash,
        db_path=db_path,
    )
    return _public_user_dict(created)


def authenticate_with_password(
    identifier: str,
    password: str,
    *,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Authenticate an existing user via email/phone and password."""
    norm_val, _ = normalize_identifier(identifier)
    if not password:
        raise ValueError("Please enter your password.")

    user_row = get_user_by_identifier(norm_val, db_path=db_path)
    if user_row is None or not verify_password(password, str(user_row["password_hash"])):
        raise ValueError("Invalid email/phone number or password.")

    return _public_user_dict(user_row)


def is_smtp_configured() -> bool:
    """Return True if outgoing SMTP mail delivery is configured in environment."""
    return bool(
        os.getenv("SMTP_HOST") and os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD")
    )


def send_otp_email(recipient_email: str, otp_code: str, user_name: str = "User") -> bool:
    """Send OTP code via SMTP if configured in environment."""
    if not is_smtp_configured():
        return False

    smtp_host = str(os.getenv("SMTP_HOST")).strip()
    smtp_user = str(os.getenv("SMTP_USER")).strip()
    smtp_pass = str(os.getenv("SMTP_PASSWORD")).strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_FROM", smtp_user)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Your Verification Code: {otp_code} - AI Finance Vault"
    msg["From"] = sender
    msg["To"] = recipient_email

    text_body = f"Hello {user_name},\n\nYour AI Finance Vault login verification code is: {otp_code}\n\nThis code expires in 5 minutes."
    safe_name = html.escape(user_name)
    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: auto; padding: 24px; border: 1px solid #334155; border-radius: 12px; background: #0f172a; color: #f8fafc;">
        <h2 style="color: #60a5fa; margin-top: 0;">AI Personal Finance Vault</h2>
        <p>Hello <b>{safe_name}</b>,</p>
        <p>Your one-time login verification code is:</p>
        <div style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #34d399; margin: 20px 0; padding: 12px; background: rgba(52, 211, 153, 0.12); text-align: center; border-radius: 8px;">
            {otp_code}
        </div>
        <p style="color: #94a3b8; font-size: 14px;">This code is valid for 5 minutes. If you did not request this login, you can safely ignore this email.</p>
    </div>
    """
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to send OTP email via SMTP: %s", exc)
        return False


def _hash_otp(identifier: str, otp_code: str) -> str:
    return hashlib.sha256(f"{identifier}:{otp_code.strip()}".encode("utf-8")).hexdigest()


def request_otp(
    identifier: str,
    *,
    ttl_minutes: int = 5,
    db_path: Optional[Union[str, Path]] = None,
) -> tuple[str, dict[str, Any]]:
    """Generate and store a 6-digit OTP challenge for an existing email or phone account.

    If SMTP credentials are configured and the identifier is an email address, the OTP
    is automatically dispatched to the recipient's inbox.
    """
    norm_val, kind = normalize_identifier(identifier)
    user_row = get_user_by_identifier(norm_val, db_path=db_path)
    if user_row is None:
        raise ValueError("No account found with that email or phone number. Please sign up first.")

    otp_code = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=max(1, ttl_minutes))
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    store_otp_challenge(
        norm_val,
        _hash_otp(norm_val, otp_code),
        expires_at,
        db_path=db_path,
    )

    if kind == "email" and is_smtp_configured():
        send_otp_email(norm_val, otp_code, str(user_row.get("full_name", "User")))

    return otp_code, _public_user_dict(user_row)


def verify_otp_login(
    identifier: str,
    otp_code: str,
    *,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Verify a 6-digit OTP code and return the authenticated user dictionary."""
    norm_val, _ = normalize_identifier(identifier)
    clean_otp = (otp_code or "").strip()
    if not clean_otp.isdigit() or len(clean_otp) != 6:
        raise ValueError("Please enter the 6-digit numeric OTP code.")

    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    valid = consume_otp_challenge(
        norm_val,
        _hash_otp(norm_val, clean_otp),
        now_iso,
        db_path=db_path,
    )
    if not valid:
        raise ValueError("Invalid or expired OTP code. Please request a new code.")

    user_row = get_user_by_identifier(norm_val, db_path=db_path)
    if user_row is None:
        raise ValueError("Account not found.")
    return _public_user_dict(user_row)


def get_or_create_demo_user(
    *,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Return or create a local demo workspace user seeded with sample transactions."""
    demo_email = "demo@finai.local"
    existing = get_user_by_identifier(demo_email, db_path=db_path)
    if existing is not None:
        user = _public_user_dict(existing)
    else:
        user = register_user(
            full_name="Demo Investor",
            identifier=demo_email,
            password=secrets.token_urlsafe(18),
            secondary_identifier="+919999999999",
            db_path=db_path,
        )

    meta = get_filter_metadata(user_id=user["id"], db_path=db_path)
    if meta["total_count"] == 0:
        seed_sample_data(user_id=user["id"], db_path=db_path)
    return user


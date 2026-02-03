"""
Input validation helpers for message classification endpoints.
"""
from __future__ import annotations

import re
from typing import Tuple

MESSAGE_MIN_LENGTH = 3
MESSAGE_MAX_LENGTH = 1000
_HTML_TAG_PATTERN = re.compile(r"[<>]")


def sanitize_input(message: str) -> str:
    """Normalize user-provided message content."""
    if message is None:
        return ""
    return message.strip()


def validate_message_input(message: str) -> Tuple[bool, str | None]:
    """
    Validate a message string for classification.

    Returns a tuple of (is_valid, error_message). error_message is None when valid.
    """
    cleaned, error_message = validate_message_text(message)
    return error_message is None, error_message


def validate_message_text(message: str) -> Tuple[str | None, str | None]:
    """
    Validate and normalize a message string for classification routes.

    Returns a tuple of (clean_message, error_message). When validation fails,
    clean_message is None and error_message is a user-friendly description.
    """
    if message is None:
        return None, "Message is required."

    cleaned = sanitize_input(message)
    if not cleaned:
        return None, "Message is required."
    if len(cleaned) < MESSAGE_MIN_LENGTH or len(cleaned) > MESSAGE_MAX_LENGTH:
        return (
            None,
            f"Message must be between {MESSAGE_MIN_LENGTH} and {MESSAGE_MAX_LENGTH} characters.",
        )
    if _HTML_TAG_PATTERN.search(cleaned):
        return None, "Message cannot contain HTML tags."

    return cleaned, None

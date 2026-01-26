"""
Input validation and sanitization utilities.
"""
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def validate_message_input(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input message.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Message cannot be empty"
    
    if len(text.strip()) < 3:
        return False, "Message must be at least 3 characters long"
    
    if len(text) > 1000:
        return False, "Message cannot exceed 1000 characters"
    
    # Check for potentially harmful content (basic check)
    if re.search(r'<script|javascript:|data:', text.lower()):
        return False, "Message contains potentially harmful content"
    
    # Check for SQL injection patterns
    sql_patterns = [
        r'union\s+select', r'drop\s+table', r'delete\s+from',
        r'insert\s+into', r'update\s+set', r'exec\s*\('
    ]
    for pattern in sql_patterns:
        if re.search(pattern, text.lower()):
            logger.warning(
                "Potential SQL injection attempt detected: %s...", text[:50]
            )
            return False, "Message contains potentially harmful content"
    
    return True, None


def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing potentially harmful characters.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text

"""Twitter text cleaning and normalization utility."""
import re
import html
from typing import Optional


def clean_tweet_text(text: Optional[str], replace_urls: bool = True, strip_mentions: bool = True) -> str:
    """
    Cleans raw customer/support tweet text:
    - Decodes HTML entities (&amp;, &lt;, etc.)
    - Removes or normalizes @mentions
    - Normalizes or strips URLs
    - Normalizes whitespace and unprintable characters
    - Preserves crucial punctuation and device version numbers (e.g. iOS 11.1)
    """
    if text is None or not isinstance(text, str):
        return ""

    # Decode HTML entities
    cleaned = html.unescape(text)

    # Normalize newlines and carriage returns
    cleaned = re.sub(r"[\r\n]+", " ", cleaned)

    # Strip or normalize @mentions
    if strip_mentions:
        cleaned = re.sub(r"@\w+", "", cleaned)

    # Handle URLs
    if replace_urls:
        cleaned = re.sub(r"https?://\S+|www\.\S+", "[URL]", cleaned)
    else:
        cleaned = re.sub(r"https?://\S+|www\.\S+", "", cleaned)

    # Remove non-standard unicode control characters
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)

    # Normalize multiple spaces and extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def is_valid_text(text: str, min_chars: int = 15, max_chars: int = 500) -> bool:
    """Checks if text meets reasonable length and informational requirements."""
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < min_chars or len(stripped) > max_chars:
        return False
    # Check if text contains at least some alphanumeric characters
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return False
    return True

"""Rule-based risk and intent keyword matchers for escalation."""
import re
from typing import List, Tuple, Optional


def check_sensitive_keywords(text: str, sensitive_keywords: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Checks if text contains high-risk terms (fraud, unauthorized charge, stolen device, legal action).
    Returns (is_sensitive, matched_keyword).
    """
    lower_text = text.lower()
    for kw in sensitive_keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, lower_text):
            return True, kw
    return False, None


def check_human_agent_request(text: str, triggers: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Checks if customer explicitly requests a human representative or supervisor.
    Returns (requested_human, matched_trigger).
    """
    lower_text = text.lower()
    for trigger in triggers:
        pattern = r"\b" + re.escape(trigger.lower()) + r"\b"
        if re.search(pattern, lower_text):
            return True, trigger
    return False, None


def is_ambiguous_or_unclear(text: str, min_words: int = 4) -> bool:
    """Detects vague one-liners like 'Help me please' or 'It broke' that lack diagnosis context."""
    words = text.strip().split()
    return len(words) < min_words

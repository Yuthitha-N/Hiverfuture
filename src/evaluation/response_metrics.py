"""Automated response quality metrics."""
import re
from typing import List, Dict, Any
import numpy as np


def compute_token_overlap(text_a: str, text_b: str) -> float:
    """Computes Jaccard word token overlap between two texts."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return float(len(intersection) / len(union))


def check_brand_tone(response: str) -> Dict[str, Any]:
    """Evaluates Apple brand tone markers in the draft response."""
    lower = response.lower()
    has_polite_opener = any(phrase in lower for phrase in ["we'd love to help", "let's", "thanks for reaching out", "we're here to help", "we'd be glad", "we understand"])
    has_dm_or_action = any(phrase in lower for phrase in ["dm", "direct message", "settings >", "support", "visit", "apple.com", "step"])
    no_forbidden_promises = not any(phrase in lower for phrase in ["guarantee refund of $", "free replacement phone", "lawsuit", "my personal email"])

    score = 1.0 if (has_polite_opener and has_dm_or_action and no_forbidden_promises) else 0.7 if has_dm_or_action else 0.4
    return {
        "brand_tone_score": score,
        "has_polite_opener": has_polite_opener,
        "has_actionable_next_step": has_dm_or_action,
        "is_safe_policy_compliant": no_forbidden_promises
    }


def evaluate_response_quality(
    customer_message: str,
    draft_response: str,
    evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes automated response metrics:
    - Groundedness (overlap with retrieved evidence)
    - Query Relevance (overlap with customer message)
    - Length Compliance (character count suitable for Twitter)
    - Brand Tone compliance
    """
    # 1. Evidence Groundedness
    evidence_text = " ".join([e.get("support_response", "") for e in evidence])
    groundedness = compute_token_overlap(draft_response, evidence_text)

    # 2. Query Relevance
    relevance = compute_token_overlap(draft_response, customer_message)

    # 3. Tone & Brand Checks
    tone_checks = check_brand_tone(draft_response)

    # 4. Length
    char_len = len(draft_response)
    is_length_valid = 20 <= char_len <= 350

    return {
        "groundedness_score": float(np.round(groundedness, 4)),
        "relevance_score": float(np.round(relevance, 4)),
        "char_length": char_len,
        "is_length_valid": is_length_valid,
        **tone_checks
    }

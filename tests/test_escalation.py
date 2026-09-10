"""Unit tests for EscalationPolicy."""
import pytest
from src.escalation.policy import EscalationPolicy


def test_escalation_on_human_request():
    policy = EscalationPolicy()
    res = policy.evaluate(
        customer_message="I want to speak with a human agent supervisor right now.",
        predicted_intent="general_inquiry_features",
        confidence=0.90,
        evidence=[{"similarity_score": 0.85}],
        context=""
    )
    assert res["decision"] == "ESCALATE"
    assert res["rule_triggered"] == "EXPLICIT_HUMAN_REQUEST"


def test_escalation_on_sensitive_keyword():
    policy = EscalationPolicy()
    res = policy.evaluate(
        customer_message="There is an unauthorized fraudulent charge on my account.",
        predicted_intent="app_store_billing_subscriptions",
        confidence=0.88,
        evidence=[{"similarity_score": 0.80}],
        context=""
    )
    assert res["decision"] == "ESCALATE"
    assert res["rule_triggered"] == "SENSITIVE_KEYWORD_DETECTED"


def test_escalation_on_low_confidence():
    policy = EscalationPolicy()
    res = policy.evaluate(
        customer_message="My gadget is not doing the thing properly.",
        predicted_intent="general_inquiry_features",
        confidence=0.42,
        evidence=[{"similarity_score": 0.70}],
        context=""
    )
    assert res["decision"] == "ESCALATE"
    assert res["rule_triggered"] == "LOW_INTENT_CONFIDENCE"


def test_auto_handle_on_clear_query():
    policy = EscalationPolicy()
    res = policy.evaluate(
        customer_message="How do I check my battery health capacity in settings?",
        predicted_intent="battery_performance",
        confidence=0.92,
        evidence=[{"similarity_score": 0.78, "support_response": "Go to Settings > Battery > Battery Health."}],
        context=""
    )
    assert res["decision"] == "AUTO_HANDLE"
    assert res["rule_triggered"] == "CONFIDENT_GROUNDED_AUTO_HANDLE"

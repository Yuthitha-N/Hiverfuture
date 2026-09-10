"""Unit tests for Response Generation and Quality Checks."""
import pytest
from src.response_generation.generator import ResponseGenerator
from src.evaluation.response_metrics import evaluate_response_quality, check_brand_tone


def test_response_generator_offline():
    generator = ResponseGenerator()
    evidence = [
        {
            "similarity_score": 0.82,
            "support_response": "We'd love to help. Go to Settings > Battery to view usage details."
        }
    ]
    reply = generator.generate_response(
        customer_message="My iPhone battery drains very fast.",
        intent="battery_performance",
        evidence=evidence
    )
    assert len(reply) > 20
    assert "battery" in reply.lower() or "settings" in reply.lower()


def test_brand_tone_check():
    good_reply = "We'd love to help with this! Please check Settings > Battery and DM us if you need more help."
    bad_reply = "Hey dude, your phone is junk, get a refund."
    
    good_res = check_brand_tone(good_reply)
    bad_res = check_brand_tone(bad_reply)

    assert good_res["brand_tone_score"] >= 0.7
    assert bad_res["brand_tone_score"] < 0.7

"""Unit tests for data preprocessing and cleaner."""
import pytest
from src.data_processing.cleaner import clean_tweet_text, is_valid_text
from src.data_processing.thread_builder import parse_conversation_thread


def test_clean_tweet_text():
    raw = "@AppleSupport Hey! My battery is draining &amp; dying at 30% https://t.co/xyz123"
    cleaned = clean_tweet_text(raw)
    assert "@AppleSupport" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "[URL]" in cleaned
    assert "battery is draining" in cleaned


def test_is_valid_text():
    assert is_valid_text("My iPhone 7 will not turn on after the iOS 11 update.") is True
    assert is_valid_text("hi") is False
    assert is_valid_text("              ") is False
    assert is_valid_text(None) is False


def test_parse_conversation_thread():
    raw_convo = (
        "Customer: My iPhone screen is black and unresponsive.\n"
        "Support: We'd love to help. Have you tried a forced restart?\n"
        "Customer: Yes, but it didn't work."
    )
    parsed = parse_conversation_thread(raw_convo)
    assert parsed is not None
    assert "screen is black" in parsed["customer_message"]
    assert "forced restart" in parsed["support_response"]
    assert parsed["num_customer_turns"] == 2

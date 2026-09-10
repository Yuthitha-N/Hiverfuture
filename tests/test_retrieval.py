"""Unit tests for semantic VectorStore and Retriever."""
import pytest
import pandas as pd
from src.retrieval.vector_store import VectorStore


def test_vector_store_indexing_and_search():
    df = pd.DataFrame([
        {
            "conversation_id": "c1",
            "customer_message": "My iPhone battery dies within 2 hours.",
            "support_response": "We'd love to help. Check Settings > Battery to see which apps use the most power.",
            "intent": "battery_performance",
            "context": "Customer: battery issue"
        },
        {
            "conversation_id": "c2",
            "customer_message": "How do I reset my Apple ID password?",
            "support_response": "You can securely reset your password by visiting iforgot.apple.com.",
            "intent": "apple_id_account_security",
            "context": "Customer: password reset"
        }
    ])

    store = VectorStore(model_name="all-MiniLM-L6-v2", device="cpu")
    store.build_index(df)

    results = store.search("Why is my battery dropping so fast?", top_k=1)
    assert len(results) == 1
    assert results[0]["intent"] == "battery_performance"
    assert results[0]["similarity_score"] > 0.40
    assert "Settings > Battery" in results[0]["support_response"]

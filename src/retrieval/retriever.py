"""Historical evidence retriever wrapper."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import get_project_root, load_yaml_config
from src.retrieval.vector_store import VectorStore
from src.data_processing.loader import load_and_preprocess_brand_data

logger = get_logger(__name__)


class HistoricalRetriever:
    """
    High-level retriever that manages index construction, loading,
    and intent-grounded evidence retrieval for response generation.
    """
    def __init__(self, config_path: str = "config/config.yaml", vector_store: Optional[VectorStore] = None):
        self.config = load_yaml_config(config_path)
        root = get_project_root()
        self.index_path = root / self.config["retrieval"]["index_path"]
        self.top_k = self.config["retrieval"].get("top_k", 3)
        self.similarity_threshold = self.config["retrieval"].get("similarity_threshold", 0.55)

        if vector_store is not None:
            self.store = vector_store
        elif self.index_path.with_suffix(".npz").exists():
            self.store = VectorStore.load(self.index_path)
        else:
            logger.info("Vector index not found on disk. Building from training corpus...")
            train_df, _ = load_and_preprocess_brand_data(config_path)
            self.store = VectorStore(
                model_name=self.config["embeddings"]["model_name"],
                device=self.config["embeddings"]["device"]
            )
            self.store.build_index(train_df)
            self.store.save(self.index_path)

    def retrieve_evidence(self, customer_message: str, predicted_intent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k historical brand resolutions for a given customer inquiry.
        """
        results = self.store.search(
            query=customer_message,
            top_k=self.top_k,
            filter_intent=predicted_intent if self.config["retrieval"].get("intent_filtering", True) else None
        )
        return results

    def get_max_similarity(self, evidence: List[Dict[str, Any]]) -> float:
        """Returns the maximum similarity score among retrieved evidence."""
        if not evidence:
            return 0.0
        return max(item.get("similarity_score", 0.0) for item in evidence)

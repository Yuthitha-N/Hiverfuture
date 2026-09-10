"""Vector Store implementation for indexing and semantic retrieval of historical AppleSupport resolutions."""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Lightweight, high-performance semantic vector index.
    Stores normalized sentence embeddings and conversation metadata,
    computing exact cosine similarity via fast BLAS matrix multiplication.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device)
        self.embeddings: Optional[np.ndarray] = None
        self.records: List[Dict[str, Any]] = []

    def build_index(
        self,
        df: pd.DataFrame,
        text_column: str = "customer_message",
        batch_size: int = 64
    ) -> "VectorStore":
        """Encodes historical customer inquiries and stores structured metadata."""
        logger.info(f"Building vector index for {len(df)} records using {self.model_name}...")
        texts = df[text_column].tolist()
        self.embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.records = []
        for idx, row in df.iterrows():
            self.records.append({
                "index": int(idx),
                "conversation_id": str(row.get("conversation_id", "")),
                "customer_message": str(row.get("customer_message", "")),
                "support_response": str(row.get("support_response", "")),
                "intent": str(row.get("intent", "")),
                "context": str(row.get("context", ""))
            })
        logger.info(f"Vector index built with shape: {self.embeddings.shape}")
        return self

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_intent: Optional[str] = None,
        intent_boost: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search for a query.
        Returns top-k most similar historical records with cosine similarity scores.
        """
        if self.embeddings is None or len(self.records) == 0:
            raise RuntimeError("VectorStore is empty. Please call build_index() or load() first.")

        query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity for normalized vectors is simply the dot product
        similarities = np.dot(self.embeddings, query_emb)

        # Apply intent boost / filter if provided
        if filter_intent:
            for i, rec in enumerate(self.records):
                if rec["intent"] == filter_intent:
                    similarities[i] += intent_boost

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            rec = dict(self.records[idx])
            rec["similarity_score"] = float(np.round(similarities[idx], 4))
            results.append(rec)

        return results

    def save(self, output_path: Union[str, Path]) -> None:
        """Saves embeddings and records to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        npz_file = output_path.with_suffix(".npz")
        json_file = output_path.with_suffix(".meta.json")

        np.savez_compressed(npz_file, embeddings=self.embeddings)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "num_records": len(self.records),
                "records": self.records
            }, f)
        logger.info(f"Saved vector index to {npz_file} and {json_file}")

    @classmethod
    def load(cls, output_path: Union[str, Path], device: str = "cpu") -> "VectorStore":
        """Loads saved vector index and metadata from disk."""
        output_path = Path(output_path)
        npz_file = output_path.with_suffix(".npz")
        json_file = output_path.with_suffix(".meta.json")

        if not npz_file.exists() or not json_file.exists():
            raise FileNotFoundError(f"Vector index files not found at {output_path}")

        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        data = np.load(npz_file)
        embeddings = data["embeddings"]

        store = cls(model_name=meta.get("model_name", "all-MiniLM-L6-v2"), device=device)
        store.embeddings = embeddings
        store.records = meta.get("records", [])
        logger.info(f"Loaded VectorStore with {len(store.records)} records from {output_path}")
        return store

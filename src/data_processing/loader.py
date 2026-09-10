"""Dataset loading and preprocessing module for AppleSupport."""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from huggingface_hub import hf_hub_download

from src.utils.logger import get_logger
from src.utils.config import get_project_root, load_yaml_config
from src.data_processing.thread_builder import parse_conversation_thread
from src.data_processing.cleaner import clean_tweet_text

logger = get_logger(__name__)


def download_or_load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Downloads or loads the raw customer support conversations dataset."""
    root = get_project_root()
    raw_dir = root / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "conversations_raw.parquet"

    if raw_file.exists():
        logger.info(f"Loading cached raw data from {raw_file}")
        return pd.read_parquet(raw_file)

    # Check local twcs.csv from extracted archive
    twcs_csv = raw_dir / "twcs" / "twcs.csv"
    if not twcs_csv.exists():
        twcs_csv = raw_dir / "twcs.csv"

    if twcs_csv.exists():
        logger.info(f"Loading raw Kaggle dataset from {twcs_csv}")
        df = pd.read_csv(twcs_csv)
        logger.info(f"Loaded {len(df)} rows from {twcs_csv}")
        return df

    logger.info(f"Fetching dataset from Hugging Face: {config['data']['hf_repo']}")
    hf_path = hf_hub_download(
        repo_id=config["data"]["hf_repo"],
        filename=config["data"]["hf_filename"],
        repo_type="dataset"
    )
    df = pd.read_parquet(hf_path)
    logger.info(f"Raw dataset loaded. Shape: {df.shape}")

    # Cache locally
    df.to_parquet(raw_file, index=False)
    logger.info(f"Cached raw data to {raw_file}")
    return df


def extract_brand_pairs(df: pd.DataFrame, brand_name: str = "AppleSupport", sample_limit: int = 10000) -> pd.DataFrame:
    """Filters data for the chosen brand, parses conversation threads, and removes noisy/duplicate records."""
    logger.info(f"Filtering dataset for brand: '{brand_name}'")
    brand_df = df[df["company"] == brand_name].copy()
    logger.info(f"Found {len(brand_df)} total conversations for {brand_name}")

    parsed_records = []
    seen_queries = set()

    for _, row in brand_df.iterrows():
        raw_convo = row.get("conversation", "")
        parsed = parse_conversation_thread(raw_convo)
        if parsed is None:
            continue

        query = parsed["customer_message"]
        # Basic deduplication on normalized query
        norm_key = query.lower()[:60]
        if norm_key in seen_queries:
            continue
        seen_queries.add(norm_key)

        parsed_records.append({
            "conversation_id": row.get("conversation_id", ""),
            "customer_message": query,
            "support_response": parsed["support_response"],
            "context": parsed["context"],
            "num_customer_turns": parsed["num_customer_turns"],
            "num_support_turns": parsed["num_support_turns"]
        })

        if len(parsed_records) >= sample_limit:
            break

    result_df = pd.DataFrame(parsed_records)
    logger.info(f"Extracted {len(result_df)} clean customer-support conversation pairs for {brand_name}")
    return result_df


def assign_intent_heuristics(df: pd.DataFrame, taxonomy: Dict[str, Any]) -> pd.DataFrame:
    """
    Labels historical AppleSupport conversations into taxonomy intents using
    keyword & semantic heuristic matching for building the retrieval & training base.
    """
    intents = taxonomy["intents"]
    assigned_intents = []

    for _, row in df.iterrows():
        text = (row["customer_message"] + " " + row["context"]).lower()
        best_intent = "general_inquiry_features"
        max_matches = 0

        for item in intents:
            name = item["intent"]
            keywords = item.get("keywords", [])
            matches = sum(1 for kw in keywords if kw.lower() in text)
            if matches > max_matches:
                max_matches = matches
                best_intent = name

        assigned_intents.append(best_intent)

    df["intent"] = assigned_intents
    return df


def load_and_preprocess_brand_data(config_path: str = "config/config.yaml", force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline to load, clean, structure, label, and split the brand data into:
    1. Training & Retrieval Knowledge Corpus
    2. Evaluation/Holdout Set
    """
    config = load_yaml_config(config_path)
    root = get_project_root()
    processed_dir = root / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / "apple_support_processed.parquet"

    if processed_file.exists() and not force_reload:
        logger.info(f"Loading processed data from {processed_file}")
        df = pd.read_parquet(processed_file)
    else:
        raw_df = download_or_load_raw_data(config)
        brand_df = extract_brand_pairs(
            raw_df,
            brand_name=config["project"]["brand"],
            sample_limit=config["data"].get("sample_size", 10000)
        )
        taxonomy_path = root / config["data"]["taxonomy_path"]
        import json
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)

        df = assign_intent_heuristics(brand_df, taxonomy)
        df.to_parquet(processed_file, index=False)
        logger.info(f"Saved processed dataset to {processed_file}")

    # Split deterministically
    seed = config["project"].get("random_seed", 42)
    np.random.seed(seed)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_pct = config["data"].get("train_split", 0.8)
    split_idx = int(len(shuffled) * train_pct)

    train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    val_df = shuffled.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"Split data into Train/Retrieval Corpus: {len(train_df)}, Validation: {len(val_df)}")
    return train_df, val_df


if __name__ == "__main__":
    train, val = load_and_preprocess_brand_data()
    print("Train sample:")
    print(train[["customer_message", "intent", "support_response"]].head(3))

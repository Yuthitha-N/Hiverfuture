"""Intent classification module containing baselines, embedding models, and evaluation routines."""
from src.intent_classification.baselines import MajorityClassifier, TFIDFLogisticClassifier
from src.intent_classification.classifier import EmbeddingClassifier
from src.intent_classification.evaluate_intents import train_and_evaluate_all_classifiers

__all__ = [
    "MajorityClassifier",
    "TFIDFLogisticClassifier",
    "EmbeddingClassifier",
    "train_and_evaluate_all_classifiers"
]

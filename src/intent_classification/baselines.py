"""Baseline models for Intent Classification."""
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from collections import Counter


class MajorityClassifier:
    """
    Baseline 1 (Trivial): Always predicts the majority class observed during training.
    """
    def __init__(self):
        self.majority_intent: str = "ios_software_update"
        self.classes_: np.ndarray = np.array([])
        self.class_priors_: Dict[str, float] = {}

    def fit(self, X: List[str], y: List[str]) -> "MajorityClassifier":
        counts = Counter(y)
        self.majority_intent = counts.most_common(1)[0][0]
        self.classes_ = np.array(sorted(list(counts.keys())))
        total = len(y)
        self.class_priors_ = {cls: counts.get(cls, 0) / total for cls in self.classes_}
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return np.array([self.majority_intent] * len(X))

    def predict_proba(self, X: List[str]) -> np.ndarray:
        prob_row = [self.class_priors_[c] for c in self.classes_]
        return np.tile(prob_row, (len(X), 1))


class TFIDFLogisticClassifier:
    """
    Baseline 2 (Simple ML): N-gram TF-IDF Vectorizer + Multinomial Logistic Regression.
    """
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2), random_state: int = 42):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
                C=1.0
            ))
        ])

    def fit(self, X: List[str], y: List[str]) -> "TFIDFLogisticClassifier":
        self.pipeline.fit(X, y)
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_

    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)

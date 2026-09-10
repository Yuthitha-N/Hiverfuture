"""Evaluation and comparison harness for intent classification models."""
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.utils.logger import get_logger
from src.utils.config import get_project_root, load_yaml_config
from src.data_processing.loader import load_and_preprocess_brand_data
from src.intent_classification.baselines import MajorityClassifier, TFIDFLogisticClassifier
from src.intent_classification.classifier import EmbeddingClassifier

logger = get_logger("evaluate_intents")


def plot_confusion_matrix(cm: np.ndarray, classes: list, title: str, save_path: Path):
    """Plots and saves a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True Intent',
        xlabel='Predicted Intent'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def evaluate_model_predictions(y_true: list, y_pred: list, model_name: str, classes: list) -> dict:
    """Computes comprehensive metrics for a classifier's predictions."""
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "model_name": model_name,
        "accuracy": float(np.round(acc, 4)),
        "macro_precision": float(np.round(macro_p, 4)),
        "macro_recall": float(np.round(macro_r, 4)),
        "macro_f1": float(np.round(macro_f1, 4)),
        "weighted_f1": float(np.round(weighted_f1, 4)),
        "per_class_metrics": {
            cls: {
                "precision": float(np.round(report_dict.get(cls, {}).get("precision", 0.0), 4)),
                "recall": float(np.round(report_dict.get(cls, {}).get("recall", 0.0), 4)),
                "f1_score": float(np.round(report_dict.get(cls, {}).get("f1-score", 0.0), 4)),
                "support": int(report_dict.get(cls, {}).get("support", 0))
            }
            for cls in classes
        },
        "confusion_matrix": cm.tolist()
    }


def train_and_evaluate_all_classifiers(config_path: str = "config/config.yaml") -> dict:
    """Trains Baseline 1, Baseline 2, and Main Model; evaluates on validation set; saves results."""
    root = get_project_root()
    config = load_yaml_config(config_path)
    
    figures_dir = root / config["evaluation"]["figures_dir"]
    metrics_dir = root / config["evaluation"]["metrics_dir"]
    outputs_dir = root / "outputs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_and_preprocess_brand_data(config_path)
    
    X_train = train_df["customer_message"].tolist()
    y_train = train_df["intent"].tolist()
    X_val = val_df["customer_message"].tolist()
    y_val = val_df["intent"].tolist()

    all_classes = sorted(list(set(y_train) | set(y_val)))
    results = {}

    # 1. Baseline 1: Majority Class
    logger.info("--- Evaluating Baseline 1: Majority Class Classifier ---")
    maj_clf = MajorityClassifier()
    maj_clf.fit(X_train, y_train)
    y_pred_maj = maj_clf.predict(X_val)
    results["baseline_1_majority"] = evaluate_model_predictions(y_val, y_pred_maj, "Baseline 1 (Majority)", all_classes)

    # 2. Baseline 2: TF-IDF + Logistic Regression
    logger.info("--- Training & Evaluating Baseline 2: TF-IDF + Logistic Regression ---")
    tfidf_clf = TFIDFLogisticClassifier(random_state=config["project"]["random_seed"])
    tfidf_clf.fit(X_train, y_train)
    y_pred_tfidf = tfidf_clf.predict(X_val)
    results["baseline_2_tfidf_lr"] = evaluate_model_predictions(y_val, y_pred_tfidf, "Baseline 2 (TF-IDF + LR)", all_classes)
    plot_confusion_matrix(
        np.array(results["baseline_2_tfidf_lr"]["confusion_matrix"]),
        all_classes,
        "Baseline 2 (TF-IDF + LR) Confusion Matrix",
        figures_dir / "cm_baseline_tfidf.png"
    )

    # 3. Main Model: Sentence-Transformers Embedding + Calibrated Logistic Regression
    logger.info("--- Training & Evaluating Main Model: Sentence-Transformers + LR ---")
    embed_clf = EmbeddingClassifier(
        model_name=config["embeddings"]["model_name"],
        device=config["embeddings"]["device"],
        random_state=config["project"]["random_seed"]
    )
    embed_clf.fit(X_train, y_train, batch_size=config["embeddings"]["batch_size"])
    y_pred_embed = embed_clf.predict(X_val, batch_size=config["embeddings"]["batch_size"])
    results["main_embedding_classifier"] = evaluate_model_predictions(y_val, y_pred_embed, "Main Model (Embedding + LR)", all_classes)
    plot_confusion_matrix(
        np.array(results["main_embedding_classifier"]["confusion_matrix"]),
        all_classes,
        "Main Model (MiniLM Embeddings + LR) Confusion Matrix",
        figures_dir / "cm_main_model.png"
    )

    # Save trained checkpoint
    model_save_path = root / config["classification"]["saved_model_path"]
    embed_clf.save(model_save_path)

    # Save metrics JSON
    metrics_path = metrics_dir / "intent_classification_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved intent classification metrics to {metrics_path}")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Model':<30} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("="*70)
    for k, v in results.items():
        print(f"{v['model_name']:<30} | {v['accuracy']:<10.4f} | {v['macro_f1']:<10.4f} | {v['weighted_f1']:<10.4f}")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    train_and_evaluate_all_classifiers()

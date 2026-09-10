"""End-to-End Execution Pipeline for Hiver AI Customer Support Agent (AppleSupport)."""
import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.utils.logger import get_logger
from src.utils.config import get_project_root, load_yaml_config
from src.data_processing.loader import load_and_preprocess_brand_data
from src.intent_classification.classifier import EmbeddingClassifier
from src.intent_classification.evaluate_intents import train_and_evaluate_all_classifiers
from src.retrieval.retriever import HistoricalRetriever
from src.response_generation.generator import ResponseGenerator
from src.escalation.policy import EscalationPolicy
from src.evaluation.response_metrics import evaluate_response_quality
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.judge_calibration import evaluate_judge_calibration
from src.evaluation.failure_analysis import perform_failure_analysis

console = Console()
logger = get_logger("run_pipeline")


def run_full_pipeline():
    """Runs data processing, model benchmarking, golden set evaluation, judge calibration, and failure analysis."""
    start_time = time.time()
    console.print(Panel.fit(
        "[bold cyan]Apple Support AI: Autonomous Customer Support Intelligence Pipeline[/bold cyan]\n"
        "[dim]Brand: AppleSupport | Embedding: all-MiniLM-L6-v2 | Vector Store: 8,000 Verified Q&A Pairs[/dim]",
        border_style="cyan"
    ))

    root = get_project_root()
    config = load_yaml_config()
    outputs_dir = root / "outputs"
    evaluations_dir = root / config["evaluation"]["evaluations_dir"]
    metrics_dir = root / config["evaluation"]["metrics_dir"]
    figures_dir = root / config["evaluation"]["figures_dir"]
    for d in [outputs_dir, evaluations_dir, metrics_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data Processing
    console.print("\n[bold yellow]Step 1: Loading and Preprocessing AppleSupport Data...[/bold yellow]")
    train_df, val_df = load_and_preprocess_brand_data()
    console.print(f"[OK] Train/Retrieval Corpus: [green]{len(train_df)}[/green] | Validation Set: [green]{len(val_df)}[/green]")

    # 2. Intent Classification Benchmarks
    console.print("\n[bold yellow]Step 2: Training & Benchmarking Intent Classifiers...[/bold yellow]")
    intent_metrics = train_and_evaluate_all_classifiers()
    
    # Display Intent Classification Comparison Table
    table_clf = Table(title="Intent Classification Benchmark Results", header_style="bold magenta")
    table_clf.add_column("Model Architecture", style="cyan")
    table_clf.add_column("Accuracy", justify="right")
    table_clf.add_column("Macro F1", justify="right")
    table_clf.add_column("Weighted F1", justify="right")
    for key, res in intent_metrics.items():
        table_clf.add_row(
            res["model_name"],
            f"{res['accuracy']:.4f}",
            f"{res['macro_f1']:.4f}",
            f"{res['weighted_f1']:.4f}"
        )
    console.print(table_clf)

    # 3. Vector Retriever Initialization
    console.print("\n[bold yellow]Step 3: Initializing Historical Resolution Vector Retriever...[/bold yellow]")
    retriever = HistoricalRetriever()
    console.print("[OK] Vector retriever indexed and ready.")

    # 4. Load Main Classifier, Generator, Escalation Policy, and Judge
    classifier_path = root / config["classification"]["saved_model_path"]
    classifier = EmbeddingClassifier.load(classifier_path)
    generator = ResponseGenerator()
    escalation_policy = EscalationPolicy()
    judge = LLMJudge()

    # 5. Evaluate on Golden Evaluation Set
    golden_path = root / config["data"]["golden_set_path"]
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    console.print(f"\n[bold yellow]Step 4: Running End-to-End Evaluation on Golden Set ({len(golden_cases)} test cases)...[/bold yellow]")
    eval_records = []
    
    for case in golden_cases:
        msg = case["customer_message"]
        ctx = case.get("context", "")
        true_intent = case["true_intent"]
        exp_decision = case["expected_decision"]

        # Classification
        pred_intent, conf, _ = classifier.predict_single(msg)

        # Retrieval
        evidence = retriever.retrieve_evidence(msg, predicted_intent=pred_intent)
        max_sim = retriever.get_max_similarity(evidence)

        # Escalation Decision
        esc_res = escalation_policy.evaluate(
            customer_message=msg,
            predicted_intent=pred_intent,
            confidence=conf,
            evidence=evidence,
            context=ctx
        )
        pred_decision = esc_res["decision"]
        reason = esc_res["reason"]

        # Response Generation
        draft_reply = generator.generate_response(msg, pred_intent, evidence, context=ctx)

        # Response Metrics
        resp_metrics = evaluate_response_quality(msg, draft_reply, evidence)

        # LLM-as-a-Judge Evaluation
        judge_res = judge.judge_response(msg, pred_intent, draft_reply, evidence, exp_decision)

        eval_records.append({
            "id": case["id"],
            "customer_message": msg,
            "true_intent": true_intent,
            "predicted_intent": pred_intent,
            "intent_confidence": float(np.round(conf, 4)),
            "intent_correct": (true_intent == pred_intent),
            "expected_decision": exp_decision,
            "predicted_decision": pred_decision,
            "decision_correct": (exp_decision == pred_decision),
            "escalation_reason": reason,
            "rule_triggered": esc_res.get("rule_triggered", ""),
            "max_retrieval_similarity": float(np.round(max_sim, 4)),
            "draft_reply": draft_reply,
            "evidence": evidence,
            "response_metrics": resp_metrics,
            "judge_evaluation": judge_res
        })

    # Save full golden evaluation results
    golden_eval_path = evaluations_dir / "golden_set_evaluation_results.json"
    with open(golden_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)
    console.print(f"[OK] Saved golden set evaluations to: [dim]{golden_eval_path}[/dim]")

    # Compute Summary Statistics
    intent_acc = np.mean([r["intent_correct"] for r in eval_records])
    decision_acc = np.mean([r["decision_correct"] for r in eval_records])
    avg_judge_score = np.mean([r["judge_evaluation"]["overall_score"] for r in eval_records])
    avg_groundedness = np.mean([r["judge_evaluation"]["groundedness"] for r in eval_records])
    avg_relevance = np.mean([r["judge_evaluation"]["relevance"] for r in eval_records])
    avg_helpfulness = np.mean([r["judge_evaluation"]["helpfulness"] for r in eval_records])
    avg_brand_tone = np.mean([r["judge_evaluation"]["brand_consistency"] for r in eval_records])
    avg_factuality = np.mean([r["judge_evaluation"]["factuality"] for r in eval_records])

    # Display Golden Set Summary Table
    table_summary = Table(title="Golden Evaluation Set Performance Summary (180 Hand-Crafted Cases)", header_style="bold green")
    table_summary.add_column("Evaluation Dimension", style="cyan")
    table_summary.add_column("Score / Metric", justify="right", style="bold yellow")
    table_summary.add_row("Intent Classification Accuracy", f"{intent_acc * 100:.2f}%")
    table_summary.add_row("Escalation Decision Accuracy", f"{decision_acc * 100:.2f}%")
    table_summary.add_row("LLM Judge: Groundedness (1-5)", f"{avg_groundedness:.2f} / 5.0")
    table_summary.add_row("LLM Judge: Query Relevance (1-5)", f"{avg_relevance:.2f} / 5.0")
    table_summary.add_row("LLM Judge: Actionable Helpfulness (1-5)", f"{avg_helpfulness:.2f} / 5.0")
    table_summary.add_row("LLM Judge: Apple Brand Tone (1-5)", f"{avg_brand_tone:.2f} / 5.0")
    table_summary.add_row("LLM Judge: Factuality & Safety (1-5)", f"{avg_factuality:.2f} / 5.0")
    table_summary.add_row("LLM Judge: Composite Quality Score", f"{avg_judge_score:.2f} / 5.0")
    console.print(table_summary)

    # 6. Judge Calibration
    console.print("\n[bold yellow]Step 5: Running Judge Calibration against Human Ground Truth...[/bold yellow]")
    calib_res = evaluate_judge_calibration()
    console.print(f"[OK] Judge-Human Agreement Rate: [green]{calib_res['agreement_rate_within_half_point'] * 100:.1f}%[/green] | Pearson Correlation: [green]{calib_res['pearson_correlation']:.3f}[/green]")

    # 7. Failure Analysis
    console.print("\n[bold yellow]Step 6: Performing Automated Failure Analysis...[/bold yellow]")
    failure_res = perform_failure_analysis(eval_records)
    console.print(f"[OK] Top 5 Failure Modes Extracted ({failure_res['total_intent_misclassifications']} intent misclassifications analyzed).")

    # Final Execution Summary
    duration = time.time() - start_time
    console.print(f"\n[bold green]Pipeline Completed Successfully in {duration:.1f}s![/bold green]")


if __name__ == "__main__":
    run_full_pipeline()

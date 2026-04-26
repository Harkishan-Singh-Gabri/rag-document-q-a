"""
Evaluation Script — The Resume Differentiator
============================================
This script runs experiments comparing different RAG configurations
and tracks results in MLflow.

What it tests:
- Different chunk sizes (256, 500, 1024)
- Different top_k values (1, 3, 5)
- Measures: ROUGE score, answer relevance, response time

How to run:
    python experiments/eval.py

Then view results:
    mlflow ui
    # Open http://localhost:5000
"""

import sys
import os
import time
import json
import mlflow
import mlflow.sklearn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouge_score import rouge_scorer
from app.rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

# ── Configure MLflow ──────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "./mlflow_tracking"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("RAG_Chunk_Size_Experiment")

# ── Sample QA pairs for evaluation ────────────────────────────────────────────
# IMPORTANT: Replace these with real Q&A pairs from YOUR test document
# Create 10-15 pairs manually by reading the document
SAMPLE_QA_PAIRS = [
    {
        "question": "What is the main purpose of the document?",
        "expected": "The document describes..."  # Fill this in with real answers
    },
    {
        "question": "What are the key findings?",
        "expected": "The key findings include..."
    },
    {
        "question": "Who are the authors?",
        "expected": "The authors are..."
    },
    # Add more Q&A pairs here based on your test document
]

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }


def is_answer_grounded(answer: str) -> bool:
    """
    Simple check: did the model say it couldn't find the answer?
    (indicates retrieval failure)
    """
    not_found_phrases = [
        "couldn't find",
        "not in the context",
        "i don't know",
        "no information",
        "not mentioned"
    ]
    return not any(phrase in answer.lower() for phrase in not_found_phrases)


def run_experiment(pdf_path: str, chunk_size: int, chunk_overlap: int, top_k: int):
    """Run one experiment configuration and log to MLflow."""

    with mlflow.start_run(run_name=f"chunk{chunk_size}_overlap{chunk_overlap}_topk{top_k}"):

        # Log parameters
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("pdf_file", os.path.basename(pdf_path))

        # Ingest document
        pipeline = RAGPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k
        )
        ingest_stats = pipeline.ingest(pdf_path)
        mlflow.log_metric("num_chunks", ingest_stats["num_chunks"])
        mlflow.log_metric("ingestion_time", ingest_stats["ingestion_time_seconds"])

        # Run evaluation
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        response_times = []
        grounded_count = 0

        results_log = []

        for qa in SAMPLE_QA_PAIRS:
            result = pipeline.query(qa["question"])
            answer = result["answer"]

            # ROUGE scores
            rouge = compute_rouge(answer, qa["expected"])
            rouge1_scores.append(rouge["rouge1"])
            rouge2_scores.append(rouge["rouge2"])
            rougeL_scores.append(rouge["rougeL"])

            # Groundedness
            if is_answer_grounded(answer):
                grounded_count += 1

            # Timing
            response_times.append(result["response_time_seconds"])

            results_log.append({
                "question": qa["question"],
                "expected": qa["expected"],
                "predicted": answer,
                "rouge1": rouge["rouge1"],
                "top_retrieval_score": result["scores"][0] if result["scores"] else 0
            })

        # Log aggregate metrics
        mlflow.log_metric("avg_rouge1", sum(rouge1_scores) / len(rouge1_scores))
        mlflow.log_metric("avg_rouge2", sum(rouge2_scores) / len(rouge2_scores))
        mlflow.log_metric("avg_rougeL", sum(rougeL_scores) / len(rougeL_scores))
        mlflow.log_metric("groundedness_rate", grounded_count / len(SAMPLE_QA_PAIRS))
        mlflow.log_metric("avg_response_time", sum(response_times) / len(response_times))

        # Save detailed results as artifact
        results_path = f"results_chunk{chunk_size}_topk{top_k}.json"
        with open(results_path, "w") as f:
            json.dump(results_log, f, indent=2)
        mlflow.log_artifact(results_path)
        os.remove(results_path)

        print(f"✅ chunk={chunk_size}, top_k={top_k} | ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.3f}")


# ── Run all experiments ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Path to your test PDF — replace with your actual file
    TEST_PDF = "data/sample_docs/test_document.pdf"

    if not os.path.exists(TEST_PDF):
        print(f"⚠️  Test PDF not found at: {TEST_PDF}")
        print("Please add a test PDF to data/sample_docs/ and update SAMPLE_QA_PAIRS")
        exit(1)

    print("🔬 Starting RAG experiments...")
    print("=" * 50)

    # Experiment grid
    chunk_sizes = [256, 500, 1024]
    top_k_values = [1, 3, 5]

    for chunk_size in chunk_sizes:
        for top_k in top_k_values:
            run_experiment(
                pdf_path=TEST_PDF,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size // 10,  # 10% overlap
                top_k=top_k
            )

    print("\n✅ All experiments complete!")
    print("Run: mlflow ui")
    print("Open: http://localhost:5000 to compare results")

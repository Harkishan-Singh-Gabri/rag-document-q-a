import sys
import os
import csv
import time
from rouge_score import rouge_scorer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_pipeline import RAGPipeline
from app.llm import LLMError
from dotenv import load_dotenv

load_dotenv()

RESULTS_CSV = "experiments/results.csv"
API_CALL_DELAY_SECONDS = 1.5 

SAMPLE_QA_PAIRS = [
    {
        "question": "What is the definition of engineering given in the lecture?",
        "expected": "Engineering is a professional discipline that combines knowledge from physics, mathematics, and material science with innovation, analysis, and design to develop technologies and solutions that meet human needs in a safe, efficient, and sustainable way.",
        "source_snippet": "Engineering is a professional discipline that combines knowledge",
    },
    {
        "question": "What are the four characteristics of engineering as a discipline?",
        "expected": "Interdisciplinary nature, a strong foundation in math and science, involvement of design/analysis/experimentation, and an emphasis on continual improvement and lifelong learning.",
        "source_snippet": "Interdisciplinary nature.",
    },
    {
        "question": "Which engineering disciplines are needed to build a self-driving car?",
        "expected": "Mechanical engineering for vehicle design, electrical engineering for sensors and control systems, computer engineering for AI and software, mathematics for path planning algorithms, and ethics/law for safety and legal compliance.",
        "source_snippet": "Mechanical Engineering → Vehicle design",
    },
    {
        "question": "Can a smartphone be designed using only Computer Engineering knowledge?",
        "expected": "No. A smartphone is an interdisciplinary product requiring computer engineering, electrical/electronics engineering, mechanical engineering, materials science and chemistry, telecommunication engineering, industrial design, mathematics, and ethics/cybersecurity.",
        "source_snippet": "No, a modern smartphone cannot be designed using only Computer Engineering knowledge",
    },
    {
        "question": "Why is mathematics important for engineers?",
        "expected": "Mathematics helps engineers perform calculations accurately, model and analyze systems, optimize designs, and make predictions.",
        "source_snippet": "Perform calculations accurately",
    },
    {
        "question": "What are the three steps involved in the engineering process described in the deck (design, analysis, experimentation)?",
        "expected": "Design is creating a solution after identifying a need; analysis is evaluating whether the design works safely, efficiently, and economically using math and simulations; experimentation is testing prototypes to validate the design and collect real-world data.",
        "source_snippet": "Design is the process of creating a solution to a problem",
    },
    {
        "question": "What examples are given for continual improvement and lifelong learning in engineering?",
        "expected": "A software engineer who initially learns C, Java, and database systems may later learn cloud computing, AI, cybersecurity, and data science; the deck also cites engineers adapting to Generative AI and LLMs as a real-life example.",
        "source_snippet": "Cloud Computing",
    },
    {
        "question": "What are the six key components of engineering as a discipline?",
        "expected": "Scientific foundation, mathematical modeling, design thinking, technical knowledge, experimentation and analysis, and innovation and research.",
        "source_snippet": "Scientific Foundation:",
    },
    {
        "question": "According to the lecture, why might a bridge collapse due to inadequately applied engineering components?",
        "expected": "Failure could stem from an inadequate scientific foundation (misunderstanding forces, stress, material behavior), incorrect mathematical modeling of load capacity and safety factors, or insufficient experimentation and analysis before construction.",
        "source_snippet": "Failure to properly understand forces, stress, and material behavior",
    },
    {
        "question": "How does the lecture define engineering as a profession?",
        "expected": "Engineering as a profession is the practical application of engineering knowledge to design, develop, operate, and maintain systems, products, and processes serving society, done responsibly, ethically, and efficiently, and it is licensed and regulated by professional bodies.",
        "source_snippet": "Engineering as a profession goes beyond classroom learning",
    },
    {
        "question": "What are the six key elements of engineering as a profession?",
        "expected": "Application of knowledge, professional responsibility, ethics and integrity, license and certification, teamwork and communication, and lifelong learning and innovation.",
        "source_snippet": "Application of Knowledge:",
    },
    {
        "question": "What professional license must a civil engineer in the US obtain to approve structural designs for public infrastructure?",
        "expected": "A Professional Engineer (PE) license, without which they cannot legally sign off on critical design documents submitted for construction approval.",
        "source_snippet": "a civil engineer must obtain a Professional Engineer (PE) license",
    },
    {
        "question": "What professional bodies are mentioned as examples that license and regulate engineers?",
        "expected": "IEEE (Institute of Electrical and Electronics Engineers), ASCE (American Society of Civil Engineers), and NEC (North Eastern Council).",
        "source_snippet": "IEEE-Institute of Electrical and Electronics Engineers",
    },
    {
        "question": "According to the comparison table, what is the main difference in focus between engineering as a discipline versus a profession?",
        "expected": "As a discipline, the focus is generating knowledge, theories, innovation, and research; as a profession, the focus is the practical application of that knowledge to solve real-world problems.",
        "source_snippet": "Generation of Knowledge, theories, innovation and Research",
    },
    {
        "question": "What should an engineer do if experimentation contradicts theoretical analysis?",
        "expected": "Recheck theoretical calculations and assumptions, verify the experimental setup and measurements, identify possible sources of error, repeat the experiment if necessary, and modify the design or model based on the findings — not ignore the discrepancy.",
        "source_snippet": "Recheck the theoretical calculations and assumptions",
    },
]


def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {k: v.fmeasure for k, v in scores.items()}


def not_abstained(answer: str) -> bool:
    """
    True if the model attempted an answer rather than declining.
    NOTE: this is an abstention-rate proxy, not a faithfulness/groundedness
    metric — it does not verify the answer is actually supported by context.
    """
    not_found_phrases = [
        "couldn't find", "not in the context", "i don't know",
        "no information", "not mentioned"
    ]
    return not any(phrase in answer.lower() for phrase in not_found_phrases)


def retrieval_hit(sources: list, source_snippet: str) -> bool:
    """True if the known answer-bearing snippet appears in any retrieved chunk."""
    if not source_snippet:
        return None  # not measurable for this QA pair
    return any(source_snippet.lower() in chunk.lower() for chunk in sources)


def run_experiment(pdf_path, chunk_size, chunk_overlap, top_k, writer):
    pipeline = RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
    ingest_stats = pipeline.ingest(pdf_path)

    rouge1, rouge2, rougeL = [], [], []
    times, abstain_flags, hit_flags = [], [], []
    errors = 0

    for qa in SAMPLE_QA_PAIRS:
        result = pipeline.query(qa["question"])
        time.sleep(API_CALL_DELAY_SECONDS)

        if result.get("error") and result.get("answer") is None:
            errors += 1
            continue

        answer = result["answer"]
        r = compute_rouge(answer, qa["expected"])
        rouge1.append(r["rouge1"]); rouge2.append(r["rouge2"]); rougeL.append(r["rougeL"])
        abstain_flags.append(not_abstained(answer))
        times.append(result["response_time_seconds"])

        hit = retrieval_hit(result["sources"], qa.get("source_snippet", ""))
        if hit is not None:
            hit_flags.append(hit)

    n = max(len(rouge1), 1)  # avoid div-by-zero if every call errored
    row = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "num_chunks": ingest_stats["num_chunks"],
        "avg_rouge1": round(sum(rouge1) / n, 3) if rouge1 else None,
        "avg_rouge2": round(sum(rouge2) / n, 3) if rouge2 else None,
        "avg_rougeL": round(sum(rougeL) / n, 3) if rougeL else None,
        "not_abstained_rate": round(sum(abstain_flags) / n, 3) if abstain_flags else None,
        "retrieval_hit_rate": round(sum(hit_flags) / len(hit_flags), 3) if hit_flags else None,
        "avg_response_time": round(sum(times) / n, 2) if times else None,
        "llm_errors": errors,
    }
    writer.writerow(row)
    print(f"chunk={chunk_size}, top_k={top_k} | ROUGE-1: {row['avg_rouge1']} | errors: {errors}")


if __name__ == "__main__":
    TEST_PDF = "data/sample_docs/test_document.pdf"

    if not os.path.exists(TEST_PDF):
        print(f"Test PDF not found at: {TEST_PDF}")
        print("Add a real PDF there before running.")
        sys.exit(1)

    if not SAMPLE_QA_PAIRS:
        print("SAMPLE_QA_PAIRS is empty. Fill it in with real Q&A pairs from your test PDF first.")
        sys.exit(1)

    os.makedirs("experiments", exist_ok=True)
    fieldnames = [
        "chunk_size", "chunk_overlap", "top_k", "num_chunks",
        "avg_rouge1", "avg_rouge2", "avg_rougeL",
        "not_abstained_rate", "retrieval_hit_rate", "avg_response_time", "llm_errors",
    ]

    print("Starting RAG experiments...")
    print("=" * 50)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk_size in [256, 500, 1024]:
            for top_k in [1, 3, 5]:
                run_experiment(TEST_PDF, chunk_size, chunk_size // 10, top_k, writer)

    print(f"\nAll experiments complete! Results saved to {RESULTS_CSV}")

import os
import time
from typing import List, Dict, Optional
from app.chunking import extract_and_chunk
from app.embeddings import embed_texts, build_faiss_index, search_index, save_index, load_index
from app.llm import get_answer


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage:
        pipeline = RAGPipeline(chunk_size=500, chunk_overlap=50, top_k=3)
        pipeline.ingest("path/to/document.pdf")
        result = pipeline.query("What is the main topic?")
        print(result["answer"])
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        llm_model: str = "mixtral-8x7b-32768"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model

        self.chunks: List[str] = []
        self.index = None
        self.ingested_file: Optional[str] = None

    def ingest(self, pdf_path: str) -> Dict:
        """
        Process a PDF: extract text → chunk → embed → build index.

        Returns:
            dict with ingestion stats
        """
        print(f"Ingesting: {pdf_path}")
        start = time.time()

        # Step 1: Extract and chunk
        self.chunks = extract_and_chunk(
            pdf_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        print(f"Created {len(self.chunks)} chunks")

        # Step 2: Embed chunks
        embeddings = embed_texts(self.chunks)

        # Step 3: Build FAISS index
        self.index = build_faiss_index(embeddings)
        self.ingested_file = os.path.basename(pdf_path)

        elapsed = time.time() - start

        return {
            "file": self.ingested_file,
            "num_chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "ingestion_time_seconds": round(elapsed, 2)
        }

    def query(self, question: str) -> Dict:
        """
        Answer a question using retrieved context.

        Returns:
            dict with answer, source chunks, retrieval scores, timing
        """
        if self.index is None or not self.chunks:
            return {
                "answer": "No document ingested. Please upload a PDF first.",
                "sources": [],
                "scores": []
            }

        start = time.time()

        # Step 1: Retrieve relevant chunks
        results = search_index(question, self.index, self.chunks, self.top_k)
        retrieved_chunks = [chunk for chunk, score in results]
        retrieval_scores = [score for chunk, score in results]

        # Step 2: Get LLM answer
        llm_response = get_answer(question, retrieved_chunks, self.llm_model)

        elapsed = time.time() - start

        return {
            "answer": llm_response["answer"],
            "sources": retrieved_chunks,
            "scores": retrieval_scores,
            "model": llm_response["model"],
            "tokens_used": llm_response["total_tokens"],
            "response_time_seconds": round(elapsed, 2),
            "top_k": self.top_k
        }

    def save(self, save_dir: str):
        """Persist index to disk."""
        save_index(self.index, self.chunks, save_dir)

    def load(self, save_dir: str):
        """Load index from disk."""
        self.index, self.chunks = load_index(save_dir)

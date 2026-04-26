import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload a PDF and ask questions about it.",
    version="1.0.0"
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
# In production you'd use session-based or user-based instances
pipeline = RAGPipeline(
    chunk_size=500,
    chunk_overlap=50,
    top_k=3
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request/Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    scores: list
    tokens_used: int
    response_time_seconds: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Q&A API is running. POST /upload then POST /ask"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": pipeline.ingested_file is not None,
        "chunks": len(pipeline.chunks)
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest into pipeline
    try:
        stats = pipeline.ingest(file_path)
        return {
            "message": "Document ingested successfully.",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """Ask a question about the uploaded document."""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = pipeline.query(request.question)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        scores=result["scores"],
        tokens_used=result.get("tokens_used", 0),
        response_time_seconds=result.get("response_time_seconds", 0)
    )


@app.post("/configure")
def configure(chunk_size: int = 500, chunk_overlap: int = 50, top_k: int = 3):
    """
    Reconfigure pipeline parameters.
    Useful for running experiments with different settings.
    """
    pipeline.chunk_size = chunk_size
    pipeline.chunk_overlap = chunk_overlap
    pipeline.top_k = top_k
    return {
        "message": "Pipeline reconfigured. Re-upload document to apply.",
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k
        }
    }

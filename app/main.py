import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload a PDF and ask questions about it.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline(
    chunk_size=500,
    chunk_overlap=50,
    top_k=3
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Request/Response Models 

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    scores: list
    tokens_used: int = 0
    response_time_seconds: float


# Endpoints 

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

    # Sanitize filename to prevent path traversal (e.g. "../../etc/passwd").
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                f.close()
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit."
                )
            f.write(chunk)

    try:
        stats = pipeline.ingest(file_path)
        return {"message": "Document ingested successfully.", "stats": stats}
    except ValueError as e:
        # Raised by chunking.py for scanned PDFs / empty extraction / no chunks
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """Ask a question about the uploaded document."""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = pipeline.query(request.question)

    # Distinguish a genuine LLM/backend failure from a normal "no answer" response.
    if result.get("error") and result.get("answer") is None:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {result['error']}")

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

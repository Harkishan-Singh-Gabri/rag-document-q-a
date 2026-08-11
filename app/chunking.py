import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

MIN_EXTRACTED_CHARS = 200  # below this, likely a scanned/image-only PDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.
    Handles multi-page PDFs. Page boundaries are NOT embedded as inline
    markers (that would pollute chunk embeddings) — pages are joined with
    a plain newline instead.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        pages.append(page.get_text("text"))

    doc.close()
    full_text = "\n".join(pages)

    if len(full_text.strip()) < MIN_EXTRACTED_CHARS:
        raise ValueError(
            "Extracted almost no text from this PDF. It may be a scanned/"
            "image-only document — OCR is not supported by this pipeline."
        )

    return full_text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks using LangChain's splitter.
    Tries to split on paragraph breaks first, then sentences, then words.

    Args:
        text: Raw text from PDF
        chunk_size: Max characters per chunk (try 256, 512, 1024 for experiments)
        chunk_overlap: Overlap between consecutive chunks (helps context continuity)

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_text(text)

    # Filter out very short chunks (usually noise)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

    if not chunks:
        raise ValueError("No usable chunks produced from this document.")

    return chunks


def extract_and_chunk(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """Convenience function: extract text and chunk in one call."""
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, chunk_size, chunk_overlap)

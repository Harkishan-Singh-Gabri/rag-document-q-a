import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.
    Handles multi-page PDFs and preserves paragraph structure.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # plain text extraction
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
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
    return chunks


def extract_and_chunk(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """Convenience function: extract text and chunk in one call."""
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, chunk_size, chunk_overlap)

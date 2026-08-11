import os
import time
import logging
from typing import List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_MODEL = "llama-3.1-8b-instant"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class LLMError(Exception):
    """Raised when the LLM call fails after retries."""
    pass


def build_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Build the RAG prompt.
    Key: tell the LLM to ONLY use the provided context.
    This prevents hallucination.
    """
    if not context_chunks:
        raise ValueError("build_prompt called with no context_chunks.")

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a precise and helpful document assistant.
Your job is to answer the user's question using ONLY the information provided in the context below.

Rules:
- If the answer is clearly in the context, answer it directly and concisely.
- If the answer is NOT in the context, respond with: "I couldn't find this information in the provided document."
- Do NOT make up information or use outside knowledge.
- Quote relevant parts of the context when useful.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    return prompt


def get_answer(
    question: str,
    context_chunks: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1  # low = more factual, less creative
) -> dict:
    """
    Send question + context to Groq LLM and get an answer.
    Retries on transient failures; raises LLMError if all retries fail
    (callers must NOT swallow this into the answer text — see main.py).

    Returns:
        dict with answer, model used, and token usage
    """
    prompt = build_prompt(question, context_chunks)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document Q&A assistant. Answer only from provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=1024,
            )

            return {
                "answer": response.choices[0].message.content.strip(),
                "model": model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)  # simple linear backoff

    raise LLMError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")

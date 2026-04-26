import os
from typing import List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
# Get free API key at: https://console.groq.com
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Available free models on Groq (as of 2024):
# - mixtral-8x7b-32768  → best quality, 32k context
# - llama3-8b-8192      → faster, lighter
# - llama3-70b-8192     → best but slower
DEFAULT_MODEL = "mixtral-8x7b-32768"


def build_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Build the RAG prompt.
    Key: tell the LLM to ONLY use the provided context.
    This prevents hallucination.
    """
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

    Returns:
        dict with answer, model used, and token usage
    """
    prompt = build_prompt(question, context_chunks)

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
        return {
            "answer": f"LLM Error: {str(e)}",
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

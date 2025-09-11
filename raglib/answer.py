from typing import List, Dict, Any
import re
import socket
import requests
from .retriever import Retriever
from .config import settings

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever

SYSTEM_PROMPT = """You are a precise assistant. Answer the question **only** using the provided sources.
Cite evidence with bracketed indices like [1], [2]. If unsure, say you don't know.
Keep it concise (3-6 sentences)."""

def draft_with_openai(question: str, hits: List[Dict[str, Any]], max_sources: int = 5) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    quotes = []
    for i, h in enumerate(hits[:max_sources]):
        src = h.get("meta", {}).get("source", f"doc_{i}")
        quotes.append(f"[{i+1}] {src}:\n{h['text'][:700]}")
    user = f"Question: {question}\n\nSources:\n" + "\n\n".join(quotes) + "\n\nAnswer:"
    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def _internet_available(timeout: float = 1.0) -> bool:
    """Best-effort check for general internet connectivity (no DNS dependency)."""
    try:
        # Try connecting to a well-known public resolver (Cloudflare) on UDP/TCP port 53
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            return True
    except Exception:
        return False

def _ollama_available(timeout: float = 1.0) -> bool:
    try:
        r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def draft_with_ollama(question: str, hits: List[Dict[str, Any]], max_sources: int = 5) -> str:
    quotes = []
    for i, h in enumerate(hits[:max_sources]):
        src = h.get("meta", {}).get("source", f"doc_{i}")
        quotes.append(f"[{i+1}] {src}:\n{h['text'][:800]}")

    system = (
        "You are a precise assistant. Answer ONLY using the provided sources.\n"
        "Cite evidence with bracketed indices like [1], [2]. If unsure, say you don't know.\n"
        "Keep it concise: 3-6 sentences. No extraneous text."
    )
    user = f"Question: {question}\n\nSources:\n" + "\n\n".join(quotes) + "\n\nAnswer:"

    try:
        resp = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0.2},
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if isinstance(data.get("message"), dict) and isinstance(data["message"].get("content"), str):
                return data["message"]["content"].strip()
            if data.get("response"):
                return str(data["response"]).strip()
            if data.get("choices"):
                # Just in case a choices-like format is returned
                ch0 = data["choices"][0]
                msg = ch0.get("message", {})
                if msg.get("content"):
                    return msg["content"].strip()
        return "I don't know."
    except Exception:
        return "I don't know."

def extractive_answer(hits: List[Dict[str, Any]], max_sources: int = 5) -> str:
    parts = []
    for i, h in enumerate(hits[:max_sources]):
        src = h.get("meta", {}).get("source", f"doc_{i}")
        parts.append(f"[{i+1}] {src}:\n{h['text']}")
    return "\n\n".join(parts)

def answer_question(question: str, top_k: int | None = None, llm: str | None = None, max_sources: int = 5) -> Dict[str, Any]:
    hits = _get_retriever().search(question, top_k=top_k)

    # Determine backend selection
    backend = (llm or settings.LLM_BACKEND or "none").lower()
    answer: str
    if backend == "auto":
        if settings.OPENAI_API_KEY and _internet_available():
            try:
                answer = draft_with_openai(question, hits, max_sources=max_sources)
            except Exception:
                # If OpenAI fails (e.g., no network), try Ollama next
                if _ollama_available():
                    answer = draft_with_ollama(question, hits, max_sources=max_sources)
                else:
                    answer = extractive_answer(hits, max_sources=max_sources)
        elif _ollama_available():
            answer = draft_with_ollama(question, hits, max_sources=max_sources)
        else:
            answer = extractive_answer(hits, max_sources=max_sources)
    elif backend == "openai":
        if settings.OPENAI_API_KEY and _internet_available():
            try:
                answer = draft_with_openai(question, hits, max_sources=max_sources)
            except Exception:
                # Fallback to Ollama if available
                if _ollama_available():
                    answer = draft_with_ollama(question, hits, max_sources=max_sources)
                else:
                    answer = extractive_answer(hits, max_sources=max_sources)
        else:
            # No API key or offline → try Ollama
            if _ollama_available():
                answer = draft_with_ollama(question, hits, max_sources=max_sources)
            else:
                answer = extractive_answer(hits, max_sources=max_sources)
    elif backend == "ollama":
        if _ollama_available():
            answer = draft_with_ollama(question, hits, max_sources=max_sources)
        else:
            # Fallback to OpenAI if possible, otherwise extractive
            if settings.OPENAI_API_KEY and _internet_available():
                try:
                    answer = draft_with_openai(question, hits, max_sources=max_sources)
                except Exception:
                    answer = extractive_answer(hits, max_sources=max_sources)
            else:
                answer = extractive_answer(hits, max_sources=max_sources)
    else:
        # 'none' or unknown → extractive
        answer = extractive_answer(hits, max_sources=max_sources)

    sentences = []
    for sent in re.split(r'(?<=[.!?])\s+', answer.strip()):
        cits = []
        for m in re.findall(r"\[(\d+)\]", sent):
            idx = int(m) - 1
            if 0 <= idx < len(hits):
                cits.append({
                    "doc_id": hits[idx].get("meta", {}).get("source", f"doc_{idx}"),
                    "start": hits[idx].get("start"),
                    "end": hits[idx].get("end"),
                })
        if sent:
            sentences.append({"text": sent, "citations": cits})

    return {
        "answer": answer,
        "sentences": sentences,
        "sources": hits,
        "confidence": None
    }

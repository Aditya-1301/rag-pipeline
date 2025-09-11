from typing import List, Dict, Any
import re
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

def extractive_answer(hits: List[Dict[str, Any]], max_sources: int = 5) -> str:
    parts = []
    for i, h in enumerate(hits[:max_sources]):
        src = h.get("meta", {}).get("source", f"doc_{i}")
        parts.append(f"[{i+1}] {src}:\n{h['text']}")
    return "\n\n".join(parts)

def answer_question(question: str, top_k: int | None = None, llm: str | None = None, max_sources: int = 5) -> Dict[str, Any]:
    hits = _get_retriever().search(question, top_k=top_k)
    if llm == "openai" and settings.OPENAI_API_KEY:
        draft = draft_with_openai(question, hits, max_sources=max_sources)
        answer = draft
    else:
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

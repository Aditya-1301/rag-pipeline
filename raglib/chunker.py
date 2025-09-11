from typing import List, Dict
import tiktoken

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Token-based chunking with overlap. Fallback to char if encoding fails."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        out = []
        i = 0
        while i < len(toks):
            window = toks[i:i+chunk_size]
            chunk = enc.decode(window)
            start_char = len(enc.decode(toks[:i]))
            end_char = start_char + len(chunk)
            out.append({"text": chunk, "start": start_char, "end": end_char})
            step = max(1, chunk_size - overlap)
            i += step
        return out
    except Exception:
        # Char fallback: 4 chars ~= 1 token heuristic
        out = []
        i = 0
        cs = chunk_size * 4
        ov = overlap * 4
        while i < len(text):
            window = text[i:i+cs]
            out.append({"text": window, "start": i, "end": i+len(window)})
            i += max(1, cs - ov)
        return out

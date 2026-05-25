"""Retrieval-based memory system.

Stores memories as flat text chunks and retrieves them
by query-conditioned word-overlap similarity search.
"""

import string


def create_retrieval_memory() -> list[str]:
    return []


def add_chunk(M: list[str], text: str) -> None:
    M.append(text)


def _tokenize(text: str) -> set[str]:
    words = text.lower().split()
    cleaned = {w.strip(string.punctuation) for w in words}
    cleaned.discard("")
    return cleaned


def retrieve(query: str, M: list[str], k: int = 3) -> list[str]:
    query_words = _tokenize(query)
    if not query_words:
        return M[:k]

    scored = []
    for i, chunk in enumerate(M):
        chunk_words = _tokenize(chunk)
        overlap = len(query_words & chunk_words)
        score = overlap / len(query_words)
        scored.append((score, i, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[:k]
    return [chunk for _, _, chunk in top_k]


def answer(query: str, M: list[str], k: int = 3) -> str:
    chunks = retrieve(query, M, k)
    context = " ".join(chunks)
    return f"Based on retrieved memories: {context}. The answer is: (LLM would answer here)"


if __name__ == "__main__":
    M = create_retrieval_memory()
    add_chunk(M, "Alice lives in Paris")
    add_chunk(M, "Bob works at Google")
    add_chunk(M, "Alice likes coffee")
    add_chunk(M, "Paris is in France")
    add_chunk(M, "Google is in Mountain View")

    print("=== Query 1: 'Where does Alice live?' (k=2) ===")
    chunks1 = retrieve("Where does Alice live?", M, k=2)
    print("Retrieved chunks:", chunks1)
    print(answer("Where does Alice live?", M, k=2))
    print()

    print("=== Query 2: 'Where does Bob work?' (k=2) ===")
    chunks2 = retrieve("Where does Bob work?", M, k=2)
    print("Retrieved chunks:", chunks2)
    print(answer("Where does Bob work?", M, k=2))

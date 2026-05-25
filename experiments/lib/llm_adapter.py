"""LLM adapter for DeepSeek Flash API.

Provides:
  - answer(query, context) -> str          # LLM answers a query given memory context
  - judge(question, answer, ground_truth) -> bool  # LLM judges correctness
  - embed(text) -> list[float]             # local embedding (no API)
"""

import os
import json
import time
import requests
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Track API calls for experiment report
_api_call_count = 0
_api_total_tokens = 0


def reset_counters():
    global _api_call_count, _api_total_tokens
    _api_call_count = 0
    _api_total_tokens = 0


def get_usage():
    return {"calls": _api_call_count, "total_tokens": _api_total_tokens}


# ═══════════════════════════════════════════════════════════════════════════
# Core API call
# ═══════════════════════════════════════════════════════════════════════════

def _call_api(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> Optional[str]:
    """Call DeepSeek API with retry logic."""
    global _api_call_count, _api_total_tokens

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                _api_call_count += 1
                usage = data.get("usage", {})
                _api_total_tokens += usage.get("total_tokens", 0)
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"  [LLM] API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"  [LLM] Request error (attempt {attempt+1}): {e}")
            time.sleep(RETRY_DELAY)

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Answer generation
# ═══════════════════════════════════════════════════════════════════════════

ANSWER_SYSTEM_PROMPT = """You are an agent answering questions based on your memory context.
Rules:
1. Answer ONLY based on the provided memory context.
2. If the context contains contradictory information, use the most recent or most reliable information.
3. Keep answers concise: one sentence max.
4. If the context has no relevant information, say "I don't have enough information to answer."
5. Do NOT use any knowledge outside the provided context."""


def answer(query: str, context: str) -> Optional[str]:
    """Generate an answer given query and memory context."""
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Memory context:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
    ]
    return _call_api(messages, temperature=0.0, max_tokens=256)


# ═══════════════════════════════════════════════════════════════════════════
# Judge
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """You are an evaluator. Given a question, a ground-truth answer, and a system answer, determine if the system answer is CORRECT or INCORRECT.

Rules:
1. The system answer is CORRECT if it conveys the same factual information as the ground truth, even if wording differs.
2. The system answer is INCORRECT if it contradicts the ground truth, contains factual errors, or says it doesn't know when the ground truth provides an answer.
3. If the system answer partially matches but misses key information, mark it INCORRECT.
4. Reply with exactly one word: CORRECT or INCORRECT."""


def judge(question: str, system_answer: str, ground_truth: str) -> tuple[bool, str]:
    """Judge whether the system answer matches the ground truth.
    Returns (is_correct: bool, raw_response: str).
    """
    if system_answer is None:
        return False, "API_ERROR"

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nGround truth: {ground_truth}\nSystem answer: {system_answer}\n\nVerdict:"},
    ]
    raw = _call_api(messages, temperature=0.0, max_tokens=256)
    if raw is None:
        return False, "API_ERROR"

    verdict = raw.strip().upper().rstrip(".,!;:")
    return verdict == "CORRECT", verdict


# ═══════════════════════════════════════════════════════════════════════════
# Semantic similarity judge (fallback, no API)
# ═══════════════════════════════════════════════════════════════════════════

_embedder = None
_embedder_available = None


def _check_embedder():
    global _embedder_available
    if _embedder_available is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder_available = True
        except (ImportError, ModuleNotFoundError):
            _embedder_available = False
    return _embedder_available


def _get_embedder():
    global _embedder
    if not _check_embedder():
        return None
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed(text: str) -> list[float]:
    """Get embedding vector for text (local, no API). Returns None if unavailable."""
    model = _get_embedder()
    if model is None:
        return None
    return model.encode(text, normalize_embeddings=True).tolist()


def semantic_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two texts (local embedding, no API).
    Falls back to simple word-overlap Jaccard if embeddings unavailable."""
    e1 = embed(text1)
    e2 = embed(text2)
    if e1 is None or e2 is None:
        # Fallback: Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 and not words2:
            return 1.0
        return round(len(words1 & words2) / len(words1 | words2), 4)
    dot = sum(a * b for a, b in zip(e1, e2))
    return round(dot, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set in environment")
        exit(1)

    print("Testing answer()...")
    ctx = "Alice lives in Paris. Bob lives in Berlin. Charlie lives in Tokyo."
    ans = answer("Where does Alice live?", ctx)
    print(f"  Q: Where does Alice live?")
    print(f"  A: {ans}")

    print("\nTesting judge()...")
    ok, verdict = judge("Where does Alice live?", ans or "", "Paris")
    print(f"  Verdict: {verdict} ({'PASS' if ok else 'FAIL'})")

    print(f"\nUsage: {get_usage()}")

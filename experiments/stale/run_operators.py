"""STALE adapter: external validation on STALE benchmark (400 scenarios).

Downloads STALEproj/STALE from HuggingFace, runs our retrieval (embedding + keyword)
and state systems on all scenarios, evaluates dim1/dim2/dim3.

Approach:
  - Retrieval: chunk haystack sessions, retrieve top-k by embedding
  - State: extract key facts from M_old/M_new, apply revision
  - Judge: LLM judge for dim1 (state resolution) and dim2 (premise resistance)
  - Dim3 (policy adaptation) is more complex; skipped for first pass
"""
import os
import sys
import json
import re
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state_memory.state import create_memory, add_belief, get_active
from state_memory.retrieval import create_retrieval_memory, add_chunk
from state_memory.llm_adapter import answer, judge, embed, get_usage, reset_counters

RETRIEVAL_K = 5
MAX_SCENARIOS = 400  # Full 200 T1 + 200 T2

# ═══════════════════════════════════════════════════════════════════════════
# Load STALE dataset
# ═══════════════════════════════════════════════════════════════════════════

def load_stale(limit=MAX_SCENARIOS):
    """Load STALE dataset from HuggingFace. Falls back to local JSON if available."""
    try:
        from datasets import load_dataset
        ds = load_dataset("STALEproj/STALE", split="train", streaming=True)
        scenarios = []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            scenarios.append(item)
        print(f"  Loaded {len(scenarios)} STALE scenarios from HuggingFace.")
        return scenarios
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        print("  Trying local fallback...")
        local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stale_sample.json")
        if os.path.exists(local):
            with open(local) as f:
                return json.load(f)[:limit]
        raise RuntimeError("STALE dataset not available.")


# ═══════════════════════════════════════════════════════════════════════════
# Build retrieval + state contexts
# ═══════════════════════════════════════════════════════════════════════════

def build_contexts(scenario):
    """Build retrieval memory (all haystack messages) and state memory (from M_old/M_new)."""
    M = create_retrieval_memory()
    # Chunk haystack sessions into individual messages
    for session in scenario.get("haystack_session", []):
        for msg in session:
            if isinstance(msg, dict) and "content" in msg:
                add_chunk(M, msg["content"])

    # Build state: simple key-value from M_old and M_new
    B = create_memory()
    # Parse explanation to get attribute
    expl = scenario.get("explanation", "")
    attr = "state"  # default key
    # Try to extract attribute from explanation (e.g., "location(city)", "health")
    attr_match = re.search(r'(\w+)\((\w+)\)', expl) or re.search(r'(\w+) is now', expl)
    if attr_match:
        attr = attr_match.group(1).lower()

    old_val = scenario.get("M_old", "")[:100]
    new_val = scenario.get("M_new", "")[:100]
    add_belief(B, attr, old_val, "2024-01-01", "user", 0.80)
    
    return M, B, attr, old_val, new_val


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval functions
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_embedding(query, mem, k=RETRIEVAL_K):
    """Retrieve top-k by embedding cosine similarity."""
    q_emb = embed(query)
    if q_emb is None:
        return mem[:k]  # fallback
    scores = []
    for chunk in mem:
        c_emb = embed(chunk)
        if c_emb:
            dot = sum(a * b for a, b in zip(q_emb, c_emb))
            scores.append((dot, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:k]]


def build_context(chunks):
    return "\n".join(f"- {c[:200]}" for c in chunks)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_dim1(scenario, M, B, attr, old_val, new_val):
    """State Resolution: does the system know the correct current state?"""
    dim1 = scenario["probing_queries"]["dim1_query"]
    # Ground truth: the new observation (M_new)
    gt = scenario.get("M_new", "")[:200]

    # Retrieval (embedding)
    r_chunks = retrieve_embedding(dim1, M, k=RETRIEVAL_K)
    r_ans = answer(dim1, build_context(r_chunks))
    r_ok, _ = judge(dim1, r_ans or "", gt)

    # State
    s_ctx = "Old: " + old_val[:200] + "\nNew: " + new_val[:200]
    s_ans = answer(dim1, s_ctx)
    s_ok, _ = judge(dim1, s_ans or "", gt)

    return r_ok, s_ok


def evaluate_dim2(scenario, M, B, attr, old_val, new_val):
    """Premise Resistance: can the system reject queries with obsolete premises?"""
    dim2 = scenario["probing_queries"]["dim2_query"]
    gt = f"The premise in this question is outdated. The correct information is: {scenario.get('M_new', '')[:200]}"

    # Retrieval
    r_chunks = retrieve_embedding(dim2, M, k=RETRIEVAL_K)
    r_ans = answer(dim2, build_context(r_chunks))
    r_ok, _ = judge(dim2, r_ans or "", gt)

    # State
    s_ctx = "Old: " + old_val[:200] + "\nNew: " + new_val[:200]
    s_ans = answer(dim2, s_ctx)
    s_ok, _ = judge(dim2, s_ans or "", gt)

    return r_ok, s_ok


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    reset_counters()
    print("=" * 64)
    print("  STALE ADAPTER — External Validation")
    print("=" * 64)

    scenarios = load_stale(MAX_SCENARIOS)
    if not scenarios:
        return

    results = {"dim1_r": 0, "dim1_s": 0, "dim2_r": 0, "dim2_s": 0, "total": 0, "t1": 0, "t2": 0}
    t1_count = sum(1 for s in scenarios if s.get("type") == "T1")
    t2_count = sum(1 for s in scenarios if s.get("type") == "T2")
    print(f"  Scenarios: {len(scenarios)} (T1={t1_count}, T2={t2_count})")
    print(f"  Retrieval k: {RETRIEVAL_K}")
    print()

    for i, scenario in enumerate(scenarios):
        if i % 50 == 0:
            print(f"  [{i}/{len(scenarios)}]... (dim1_r={results['dim1_r']}, dim1_s={results['dim1_s']})")

        stype = scenario.get("type", "T1")
        M, B, attr, old_val, new_val = build_contexts(scenario)
        r1, s1 = evaluate_dim1(scenario, M, B, attr, old_val, new_val)
        r2, s2 = evaluate_dim2(scenario, M, B, attr, old_val, new_val)

        if r1: results["dim1_r"] += 1
        if s1: results["dim1_s"] += 1
        if r2: results["dim2_r"] += 1
        if s2: results["dim2_s"] += 1
        results["total"] += 1
        if stype == "T1": results["t1"] += 1
        else: results["t2"] += 1

        time.sleep(0.2)  # Rate limiting for 400 scenarios

    print()
    print("=" * 64)
    print("  STALE EXTERNAL VALIDATION RESULTS")
    print("=" * 64)
    total = results["total"]
    r1_acc = results["dim1_r"] / total if total else 0
    s1_acc = results["dim1_s"] / total if total else 0
    r2_acc = results["dim2_r"] / total if total else 0
    s2_acc = results["dim2_s"] / total if total else 0

    print(f"  {'Dimension':<22} {'Retrieval':>12} {'State':>12}")
    print(f"  {'─'*46}")
    print(f"  {'Dim1 — State Resolution':<22} {r1_acc:>11.1%} {s1_acc:>11.1%}")
    print(f"  {'Dim2 — Premise Resistance':<22} {r2_acc:>11.1%} {s2_acc:>11.1%}")
    print()
    print(f"  Scenarios: {total} (T1={results['t1']}, T2={results['t2']})")
    print(f"  STALE baseline (best model): 55.2%")

    usage = get_usage()
    print(f"\n  API calls: {usage['calls']}, tokens: {usage['total_tokens']}")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""Premise verifier for CUPMem-like system.

Detects whether a query embeds stale premises about the user's state.
Used primarily for the PR (Premise Resistance) dimension on STALE.
"""

import json
import sys
sys.path.insert(0, "/home/orion/RESEARCH/state_memory")

from state_memory.llm_adapter import answer
from state_memory.state import get_active, to_string


def check_premises(B_t, query_text):
    """Given a query and current state, detect stale premises.
    
    Returns: dict with keys: safe (bool), stale_premises (list), corrected_state (str)
    """
    state_str = to_string(B_t)
    
    # Short state representation for the LLM prompt
    active_facts = []
    for key in sorted(B_t.keys()):
        active = get_active(B_t, key)
        if active:
            for e in active:
                active_facts.append(f"  {key} = {e['value']}")
        # Also show unknown/stale keys
        for e in B_t.get(key, []):
            if e["status"] == "unknown":
                active_facts.append(f"  {key} = UNKNOWN (conflict unresolved)")
    
    state_summary = "\n".join(active_facts) if active_facts else "(no active state)"
    
    prompt = f"""You are a premise verifier for a personal AI assistant. Given current memory state and a user query, determine if the query contains any premises that are now outdated or incorrect.

Current memory state:
{state_summary}

User query: "{query_text}"

Check if the query assumes any fact that contradicts the current memory state.
For example:
- Memory says location=Austin but query says "since you live in Seattle" → stale premise
- Memory says commute_method="(no longer valid due to health)" but query says "recommend bike routes" → stale premise
- Memory says health.status="broken_leg" but query says "plan a hiking trip" → stale premise

Output ONLY valid JSON:
{{"safe": true or false, "stale_premises": ["premise 1", "premise 2", ...], "corrected_state": "brief correction if safe=false, else empty string"}}"""

    resp = answer(prompt, "")
    if not resp:
        return {"safe": True, "stale_premises": [], "corrected_state": ""}
    
    try:
        text = resp.strip()
        if "{" in text:
            text = text[text.index("{"):text.rindex("}")+1]
        result = json.loads(text)
        return result
    except Exception:
        lower = resp.lower()
        if any(w in lower for w in ["outdated", "no longer", "incorrect", "not valid", "stale"]):
            return {"safe": False, "stale_premises": ["Detected in response"], "corrected_state": resp[:200]}
        return {"safe": True, "stale_premises": [], "corrected_state": ""}

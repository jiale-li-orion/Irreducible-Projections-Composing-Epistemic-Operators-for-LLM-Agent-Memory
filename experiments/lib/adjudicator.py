"""LLM-based write-time adjudicator for CUPMem-like state management.

Replaces confidence-thresholded U with LLM judgment:
  KEEP    → reject new evidence, keep old
  REPLACE → accept new evidence, deprecate old
  UNKNOWN → mark as uncertain
"""

import json
import sys
sys.path.insert(0, "/home/orion/RESEARCH/state_memory")

from state_memory.llm_adapter import answer


def adjudicate(key, old_value, new_value, evidence_text, explanation=""):
    """LLM decides KEEP / REPLACE / UNKNOWN for conflicting evidence.
    
    Returns: dict with keys: decision, reason, confidence
    """
    prompt = f"""You are a memory adjudicator for a personal AI assistant. Given existing memory and new evidence, decide whether the new evidence replaces the old memory.

Current memory:
  key: "{key}"
  current value: "{old_value[:200]}"

New evidence: "{evidence_text[:300]}"
{("Context: " + explanation[:200]) if explanation else ""}

Rules:
- KEEP: The new evidence is just additional detail, not contradictory. OR the evidence is weak/unreliable.
- REPLACE: The new evidence clearly updates or contradicts the current value. The current value should be considered outdated.
- UNKNOWN: The relationship between old and new is unclear. Neither is clearly correct.

Output ONLY valid JSON:
{{"decision": "KEEP" or "REPLACE" or "UNKNOWN", "reason": "brief reason", "confidence": 0.0-1.0}}"""

    resp = answer(prompt, "")
    if not resp:
        return {"decision": "UNKNOWN", "reason": "LLM returned empty", "confidence": 0.0}
    
    try:
        # Try to extract JSON from response
        text = resp.strip()
        if "{" in text:
            text = text[text.index("{"):text.rindex("}")+1]
        result = json.loads(text)
        if result.get("decision") not in ("KEEP", "REPLACE", "UNKNOWN"):
            return {"decision": "UNKNOWN", "reason": "Invalid decision", "confidence": 0.0}
        return result
    except Exception:
        # Fallback: if "replace" or "keep" in response text
        lower = resp.lower()
        if "replace" in lower or "yes" in lower:
            return {"decision": "REPLACE", "reason": "Parsed from response", "confidence": 0.7}
        elif "keep" in lower or "no" in lower:
            return {"decision": "KEEP", "reason": "Parsed from response", "confidence": 0.7}
        return {"decision": "UNKNOWN", "reason": "Parse failed", "confidence": 0.0}


def adjudicate_propagation(trigger_key, trigger_value, affected_key, affected_value):
    """Decide if a change in trigger_key invalidates the value of affected_key."""
    prompt = f"""You are a memory propagation adjudicator. A change in one attribute may affect another.

Trigger: {trigger_key} changed to "{trigger_value[:200]}"
Affected attribute: {affected_key} currently has value "{affected_value[:200]}"

Does the change in {trigger_key} make the current value of {affected_key} ("{affected_value[:100]}") invalid?

Examples:
- location changes (Seattle→Austin) → commute_method ("biking") may be invalid (different city layout)
- health changes (broken_leg) → commute_method ("biking") IS invalid
- weather changes (desert climate) → location_assumption ("Portland") may be invalid

Output ONLY valid JSON:
{{"decision": "REPLACE" or "KEEP" or "UNKNOWN", "reason": "brief reason", "new_value": "suggested new value if REPLACE, else empty"}}"""

    resp = answer(prompt, "")
    if not resp:
        return {"decision": "UNKNOWN", "reason": "LLM empty", "new_value": ""}
    
    try:
        text = resp.strip()
        if "{" in text:
            text = text[text.index("{"):text.rindex("}")+1]
        result = json.loads(text)
        if result.get("decision") not in ("REPLACE", "KEEP", "UNKNOWN"):
            return {"decision": "UNKNOWN", "reason": "Invalid", "new_value": ""}
        return result
    except Exception:
        lower = resp.lower()
        if "replace" in lower:
            return {"decision": "REPLACE", "reason": "Parsed", "new_value": ""}
        elif "keep" in lower:
            return {"decision": "KEEP", "reason": "Parsed", "new_value": ""}
        return {"decision": "UNKNOWN", "reason": "Parse failed", "new_value": ""}

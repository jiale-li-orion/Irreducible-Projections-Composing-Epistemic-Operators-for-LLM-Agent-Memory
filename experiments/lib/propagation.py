"""Schema-based propagation for CUPMem-like state management.

When a key is updated via REPLACE, check PROPAGATION_SCHEMA for
affected keys and adjudicate each one.
"""

import sys
sys.path.insert(0, "/home/orion/RESEARCH/state_memory")

from state_memory.state import get_active, add_belief
from state_memory.adjudicator import adjudicate_propagation

# Pre-defined dependency chains for common attribute types.
# key → [affected_keys]
PROPAGATION_SCHEMA = {
    "location.city": [
        "lifestyle.commute_method",
        "routine.transport_mode",
    ],
    "location.country": [
        "lifestyle.commute_method",
    ],
    "health.status": [
        "lifestyle.commute_method",
        "routine.activity_level",
        "routine.transport_mode",
    ],
    "weather.climate_zone": [
        "location.inferred_region",
    ],
}

# Also catch dynamic keys that contain these substrings
SCHEMA_KEYWORDS = {
    "location": ["lifestyle.commute_method", "routine.transport_mode"],
    "health": ["lifestyle.commute_method", "routine.activity_level"],
    "weather": ["location.inferred_region"],
}


def find_affected_keys(trigger_key):
    """Find keys that might be affected by a change to trigger_key."""
    # Direct match
    if trigger_key in PROPAGATION_SCHEMA:
        return PROPAGATION_SCHEMA[trigger_key]
    # Keyword match
    for kw, affected in SCHEMA_KEYWORDS.items():
        if kw in trigger_key.lower():
            return affected
    return []


def propagate(B_t, trigger_key, trigger_value):
    """After adjudication, check schema and propagate to affected keys.
    
    B_t: the memory state dict
    trigger_key: the key that was updated
    trigger_value: the new value
    
    Returns: list of (affected_key, decision) tuples
    """
    affected_keys = find_affected_keys(trigger_key)
    results = []
    
    for affected_key in affected_keys:
        active = get_active(B_t, affected_key)
        if not active:
            continue
        old_val = active[0]["value"]
        
        # Skip if same entity-referenced (e.g., location.city vs location.country both referring same person)
        result = adjudicate_propagation(trigger_key, trigger_value, affected_key, old_val)
        decision = result.get("decision", "KEEP")
        
        if decision == "REPLACE":
            # Mark old as deprecated
            for entry in active:
                entry["status"] = "deprecated"
            new_val = result.get("new_value", "").strip()
            if new_val:
                add_belief(B_t, affected_key, new_val, "NOW", "propagation", 0.65)
            else:
                add_belief(B_t, affected_key, f"(no longer valid due to {trigger_key}={trigger_value})",
                          "NOW", "propagation", 0.60)
            results.append((affected_key, "REPLACE"))
        elif decision == "UNKNOWN":
            for entry in active:
                entry["status"] = "unknown"
            results.append((affected_key, "UNKNOWN"))
        else:
            results.append((affected_key, "KEEP"))
    
    return results

"""Revision operator implementation.

U(B_t, e_t): transforms memory state B_t given new evidence e_t.
Follows AGM-like principles: EXPAND, REVISE (by confidence), minimal change.
"""

from typing import Any


def revise(
    B_t: dict[str, list[dict[str, Any]]],
    evidence: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    key = evidence["key"]
    value = evidence["value"]
    timestamp = evidence["timestamp"]
    confidence = evidence["confidence"]

    new_entry = {
        "value": value,
        "timestamp": timestamp,
        "source": evidence["source"],
        "confidence": confidence,
        "status": "active",
    }

    entries = B_t.get(key, [])
    active_entries = [e for e in entries if e["status"] == "active"]

    if not active_entries:
        if key not in B_t:
            B_t[key] = []
        B_t[key].append(new_entry)
        return B_t

    active = active_entries[0]

    if active["value"] == value:
        return B_t

    if confidence > active["confidence"]:
        active["status"] = "deprecated"
        B_t[key].append(new_entry)
    elif confidence < active["confidence"]:
        pass
    else:
        if timestamp > active["timestamp"]:
            active["status"] = "deprecated"
            B_t[key].append(new_entry)

    return B_t


if __name__ == "__main__":
    B_t: dict[str, list[dict[str, Any]]] = {
        "user_location": [
            {
                "value": "Seattle",
                "timestamp": "2024-01-01",
                "source": "user_input",
                "confidence": 0.8,
                "status": "active",
            }
        ]
    }

    def _cr(state: dict[str, list[dict[str, Any]]]) -> int:
        return sum(
            1
            for entries in state.values()
            if sum(1 for e in entries if e["status"] == "active") > 1
        )

    print("=== Initial State ===")
    print(f"CR: {_cr(B_t)}")
    print()

    # Test 1: EXPAND
    print("=== Test 1: EXPAND (new key 'hobby') ===")
    revise(
        B_t,
        {
            "key": "hobby",
            "value": "hiking",
            "timestamp": "2024-02-01",
            "source": "user_input",
            "confidence": 0.7,
        },
    )
    print(f"CR: {_cr(B_t)}")
    assert _cr(B_t) == 0
    hobby_active = [e for e in B_t["hobby"] if e["status"] == "active"]
    assert len(hobby_active) == 1
    assert hobby_active[0]["value"] == "hiking"
    print("PASS: hobby=hiking active, CR=0")
    print()

    # Test 2: NO CHANGE
    print("=== Test 2: NO CHANGE (same value 'Seattle') ===")
    revise(
        B_t,
        {
            "key": "user_location",
            "value": "Seattle",
            "timestamp": "2024-02-01",
            "source": "user_input",
            "confidence": 0.8,
        },
    )
    print(f"CR: {_cr(B_t)}")
    assert _cr(B_t) == 0
    loc_active = [e for e in B_t["user_location"] if e["status"] == "active"]
    assert len(loc_active) == 1
    assert loc_active[0]["value"] == "Seattle"
    print("PASS: user_location still Seattle active, CR=0")
    print()

    # Test 3: REVISE (new wins)
    print("=== Test 3: REVISE (new wins, Portland confidence 0.95 > 0.8) ===")
    revise(
        B_t,
        {
            "key": "user_location",
            "value": "Portland",
            "timestamp": "2024-06-01",
            "source": "user_input",
            "confidence": 0.95,
        },
    )
    print(f"CR: {_cr(B_t)}")
    assert _cr(B_t) == 0
    loc_all = B_t["user_location"]
    loc_active = [e for e in loc_all if e["status"] == "active"]
    loc_deprecated = [e for e in loc_all if e["status"] == "deprecated"]
    assert len(loc_active) == 1
    assert loc_active[0]["value"] == "Portland"
    assert len(loc_deprecated) == 1
    assert loc_deprecated[0]["value"] == "Seattle"
    print("PASS: Seattle deprecated, Portland active, CR=0")
    print()

    # Test 4: REVISE (old wins)
    print("=== Test 4: REVISE (old wins, London confidence 0.5 < 0.95) ===")
    revise(
        B_t,
        {
            "key": "user_location",
            "value": "London",
            "timestamp": "2024-07-01",
            "source": "user_input",
            "confidence": 0.5,
        },
    )
    print(f"CR: {_cr(B_t)}")
    assert _cr(B_t) == 0
    loc_active = [e for e in B_t["user_location"] if e["status"] == "active"]
    assert len(loc_active) == 1
    assert loc_active[0]["value"] == "Portland"
    london_entries = [e for e in B_t["user_location"] if e["value"] == "London"]
    assert len(london_entries) == 0
    print("PASS: London rejected, Portland still active, CR=0")
    print()

    print("=== All Tests Passed ===")

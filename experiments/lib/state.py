"""Memory state implementation.

B_t is a dict mapping keys to lists of belief entries.
Each belief entry: {value, timestamp, source, confidence, status}
status: "active" | "deprecated"
"""

from typing import Any


def create_memory() -> dict[str, list[dict[str, Any]]]:
    return {}


def add_belief(
    B_t: dict[str, list[dict[str, Any]]],
    key: str,
    value: Any,
    timestamp: str,
    source: str,
    confidence: float,
) -> None:
    entry = {
        "value": value,
        "timestamp": timestamp,
        "source": source,
        "confidence": confidence,
        "status": "active",
    }
    if key not in B_t:
        B_t[key] = []
    B_t[key].append(entry)


def get_active(B_t: dict[str, list[dict[str, Any]]], key: str) -> list[dict[str, Any]]:
    return [e for e in B_t.get(key, []) if e["status"] == "active"]


def get_all(B_t: dict[str, list[dict[str, Any]]], key: str) -> list[dict[str, Any]]:
    return list(B_t.get(key, []))


def to_string(B_t: dict[str, list[dict[str, Any]]]) -> str:
    if not B_t:
        return "B_t = {}"
    lines = ["B_t = {"]
    entries = list(B_t.items())
    for i, (key, beliefs) in enumerate(entries):
        lines.append(f'  "{key}": [')
        for j, b in enumerate(beliefs):
            comma = "," if j < len(beliefs) - 1 else ""
            lines.append(
                f'    {{"value": {b["value"]!r}, '
                f'"timestamp": "{b["timestamp"]}", '
                f'"source": "{b["source"]}", '
                f'"confidence": {b["confidence"]}, '
                f'"status": "{b["status"]}"'
                f"}}{comma}"
            )
        closing = "]," if i < len(entries) - 1 else "]"
        lines.append(f"  {closing}")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    B_t = create_memory()

    add_belief(
        B_t,
        key="user_location",
        value="Seattle",
        timestamp="2024-01-01",
        source="user_input",
        confidence=0.8,
    )
    add_belief(
        B_t,
        key="user_location",
        value="Portland",
        timestamp="2024-06-01",
        source="user_input",
        confidence=0.9,
    )
    add_belief(
        B_t,
        key="user_age",
        value=30,
        timestamp="2024-01-01",
        source="user_input",
        confidence=0.9,
    )

    print("=== get_active('user_location') ===")
    print(get_active(B_t, "user_location"))
    print()

    print("=== get_all('user_location') ===")
    print(get_all(B_t, "user_location"))
    print()

    print("=== to_string(B_t) ===")
    print(to_string(B_t))

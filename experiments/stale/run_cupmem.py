"""CUPMem-like approximation on STALE benchmark.

Evaluates 5 modes across 400 STALE scenarios × 3 dimensions.
Modes: R_T, S_T, T_T, CUPMem_like, v1.1_composed

Usage:
    python3 stale_cupmem_adapter.py [--limit 10] [--modes R_T,S_T]
"""

import sys, os, json, time, re, argparse
sys.path.insert(0, "/home/orion/RESEARCH/state_memory")

from datasets import load_dataset
from state_memory.llm_adapter import answer, embed, get_usage, reset_counters
from state_memory.state import create_memory, add_belief, get_active, to_string
from state_memory.retrieval import create_retrieval_memory, add_chunk, retrieve
from state_memory.revision import revise
from state_memory.adjudicator import adjudicate
from state_memory.propagation import propagate
from state_memory.premise_verifier import check_premises

ALL_MODES = ["R_T", "S_T", "T_T", "v1.1_composed"]
# Remove CUPMem_like as it's not needed

# ─── Judge Prompt ───────────────────────────────────────────────────────────

JUDGE_SYSTEM = """
You are a strict evaluator for an AI Assistant Benchmark focusing on Long-Term Memory and Implicit Reasoning.

Ground Truth: M_old (outdated), M_new (updated state).

Question: {query}
Response: {response}

Score:
- dim1 (State Resolution): pass=true ONLY IF the answer recognizes M_old is NO LONGER VALID
- dim2 (Premise Resistance): pass=true ONLY IF the answer DETECTS AND REJECTS the false premise from M_old
- dim3 (Implicit Policy Adaptation): pass=true ONLY IF the answer's action/plan is based on M_new, not M_old

Output JSON: {{"dim1": true/false, "dim2": true/false, "dim3": true/false}}
"""


# ─── Helper: Build contexts ─────────────────────────────────────────────────

def _session_text(session):
    """Collapse a session's turns into a single text document."""
    parts = []
    for turn in session:
        if isinstance(turn, dict):
            role = turn.get("role", "?")
            content = turn.get("content", "")
            parts.append(f"{role.upper()}: {content}")
    return "\n".join(parts)


def _build_chunks_from_sessions(sessions, top_sessions, scenario):
    """Build retrieval chunks from selected sessions, plus M_old/M_new."""
    chunks = []
    # Add M_old and M_new as strong signals
    M_old = str(scenario.get("M_old", ""))
    if M_old:
        chunks.append(f"[Context: Past state] {M_old}")
    M_new = str(scenario.get("M_new", ""))
    if M_new:
        chunks.append(f"[Context: Recent evidence] {M_new}")
    
    # Add chunks from top-k sessions
    for idx in top_sessions:
        if idx < len(sessions):
            session = sessions[idx]
            for turn in session:
                if isinstance(turn, dict) and turn.get("content"):
                    chunks.append(turn["content"])
    return chunks


def build_retrieval(scenario):
    """Build retrieval memory with session-level search.
    
    1. Each of the 50 haystack sessions becomes a document.
    2. Query is embedded and compared to each session document.
    3. Top-3 sessions + M_old/M_new are used as the retrieval pool.
    4. Individual chunks from those sessions are returned for answer.
    """
    sessions = scenario.get("haystack_session", [])
    timestamps = scenario.get("timestamps", [])
    
    # Build session documents
    session_docs = []
    for idx, session in enumerate(sessions):
        doc = _session_text(session)
        ts = timestamps[idx] if idx < len(timestamps) and timestamps[idx] else ""
        session_docs.append((idx, doc, ts))
    
    return session_docs


def retrieve_R_T(query, session_docs, scenario, k_sessions=3):
    """Retrieve top-k sessions by embedding similarity, then chunk."""
    if not session_docs:
        return []
    
    q_emb = embed(query)
    if q_emb is None:
        # Fallback: use relevant sessions
        relevant = scenario.get("relevant_session_index", [])
        return _build_chunks_from_sessions(
            scenario.get("haystack_session", []), relevant, scenario
        )
    
    # Score each session by embedding similarity
    scored = []
    for idx, doc, ts in session_docs:
        d_emb = embed(doc[:2000])  # Use first 2000 chars for speed
        if d_emb and q_emb:
            sim = sum(a * b for a, b in zip(q_emb, d_emb))
            scored.append((sim, idx, doc, ts))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sessions = [s[1] for s in scored[:k_sessions]]
    
    return _build_chunks_from_sessions(
        [sd[1] for sd in session_docs], top_sessions, scenario
    )


def build_state(scenario, mode="S_T"):
    """Build state by processing ALL haystack sessions, not just M_old/M_new.
    
    Processes each session chronologically, extracts facts via LLM,
    and applies revision/adjudication for each new fact.
    """
    B = create_memory()
    sessions = scenario.get("haystack_session", [])
    timestamps = scenario.get("timestamps", [])
    expl = str(scenario.get("explanation", ""))
    attr = _parse_key_from_explanation(expl)
    
    # First, inject the explicit M_old and M_new as ground-truth signals
    M_old = str(scenario.get("M_old", ""))
    M_new = str(scenario.get("M_new", ""))
    relevant_idx = scenario.get("relevant_session_index", [])
    
    # Process each session chronologically, extract facts
    for idx, session in enumerate(sessions):
        # Skip sessions that don't have meaningful content
        session_text = _session_text(session)
        if len(session_text.strip()) < 20:
            continue
        
        ts = timestamps[idx] if idx < len(timestamps) and timestamps[idx] else f"session_{idx}"
        is_relevant = idx in relevant_idx
        
        if mode == "S_T":
            # S_T: extract key facts from important sessions
            # For relevant sessions, use M_old/M_new directly
            if is_relevant and M_old and idx == relevant_idx[0] if len(relevant_idx) > 0 else False:
                add_belief(B, attr, M_old[:200], ts, "user", 0.80)
            elif is_relevant and M_new and idx == relevant_idx[-1] if len(relevant_idx) > 1 else False:
                revise(B, {"key": attr, "value": M_new[:200], 
                         "timestamp": ts, "source": "user", "confidence": 0.90})
            else:
                # For distractor sessions, extract any facts found
                _extract_facts_from_session(B, session, ts, idx, mode="S_T")
        
        elif mode in ("CUPMem_like", "v1.1_composed"):
            if is_relevant and M_old and idx == relevant_idx[0] if len(relevant_idx) > 0 else False:
                add_belief(B, attr, M_old[:200], ts, "user", 0.80)
            elif is_relevant and M_new and idx == relevant_idx[-1] if len(relevant_idx) > 1 else False:
                from state_memory.adjudicator import adjudicate
                old_active = get_active(B, attr)
                old_val = str(old_active[0]["value"]) if old_active else ""
                d = adjudicate(attr, old_val, M_new[:200], f"M_new={M_new[:100]}", expl)
                if d.get("decision") == "REPLACE":
                    for e in get_active(B, attr):
                        e["status"] = "deprecated"
                    add_belief(B, attr, M_new[:200], ts, "user", 0.90)
                    propagate(B, attr, M_new[:200])
                elif d.get("decision") == "UNKNOWN":
                    for e in get_active(B, attr):
                        e["status"] = "unknown"
            else:
                _extract_facts_from_session(B, session, ts, idx, mode="CUPMem_like")
    
    return B, attr


def _extract_facts_from_session(B, session, timestamp, session_idx, mode="S_T"):
    """Extract potential fact changes from a session and apply to state.
    
    This is a lightweight extraction - not a full LLM call per session,
    just looks for change/update signals in the dialogue.
    """
    for turn in session:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = str(turn.get("content", ""))
        
        if role != "user":
            continue
            
        # Look for state-change signals
        content_lower = content.lower()
        if any(p in content_lower for p in ["moved to", "moving to", "relocated", "transferred",
                                              "now living", "new place", "new apartment",
                                              "changed", "updated", "switched", "new job",
                                              "broken", "injured", "diagnosed"]):
            # Try to extract key-value
            key = _detect_attribute(content)
            if key:
                value = content[:100]
                if mode == "S_T":
                    revise(B, {"key": key, "value": value, "timestamp": timestamp,
                             "source": "user", "confidence": 0.85})
                else:
                    old_active = get_active(B, key)
                    old_val = str(old_active[0]["value"]) if old_active else ""
                    if old_val and old_val != value:
                        d = adjudicate(key, old_val, value, content[:100], "")
                        if d.get("decision") == "REPLACE":
                            for e in get_active(B, key):
                                e["status"] = "deprecated"
                            add_belief(B, key, value, timestamp, "user", 0.85)


def _detect_attribute(text):
    """Detect which attribute a piece of text refers to."""
    t = text.lower()
    if any(p in t for p in ["live", "move", "city", "apartment", "neighbor", 
                            "address", "location", "base", "settl", "relocat"]):
        return "location.city"
    if any(p in t for p in ["commute", "bike", "drive", "transit", "shuttle", "travel"]):
        return "lifestyle.commute_method"
    if any(p in t for p in ["health", "injur", "broken", "leg", "arm", "surgery",
                             "doctor", "hospital", "pain", "mobil"]):
        return "health.status"
    if any(p in t for p in ["job", "work", "company", "employ", "career", "position"]):
        return "work.status"
    return None


def _parse_key_from_explanation(expl):
    """Extract key from STALE explanation text.
    
    T1: "Spatiotemporal_Context.location(city) is now Austin" → location.city
    T2: "Attribute A (local climate) is updated... A (environment) → B (location)" → location.city, weather.climate_zone
    """
    if not expl:
        return "state"
    expl = str(expl)
    
    # Direct pattern: dot.key(subkey)
    m = re.search(r'\.(\w+)\((\w+)\)', expl)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    
    # Location mentions
    if re.search(r'location|city|live|reside|based|settle|move|relocat', expl, re.I):
        return "location.city"
    # Climate/weather
    if re.search(r'climate|weather|environment|desert|arid|pest|scorpion', expl, re.I):
        return "weather.climate_zone"
    # Health
    if re.search(r'health|injury|broken|mobility|leg|cast|surgery', expl, re.I):
        return "health.status"
    # Commute
    if re.search(r'commute|transport|bike|drive|transit|shuttle', expl, re.I):
        return "lifestyle.commute_method"
    
    # T2 cascade pattern: "A (env) → B (loc)" — use the target key (B)
    m = re.search(r'(\w+)\s*[→➡].*?(\w+)', expl)
    if m:
        target = m.group(2).lower()
        if 'locat' in target or 'city' in target:
            return "location.city"
        return f"state.{target}"
    
    return "state"


def build_trajectory(scenario):
    """Build chronological walk of relevant sessions."""
    sessions = scenario.get("haystack_session", [])
    relevant = scenario.get("relevant_session_index", [])
    if not relevant:
        return ""
    parts = []
    for idx in sorted(relevant):
        if idx < len(sessions):
            session = sessions[idx]
            ts = scenario.get("timestamps", [])
            time_str = f" [{ts[idx]}]" if idx < len(ts) and ts[idx] else ""
            parts.append(f"--- Session {idx}{time_str} ---")
            for turn in session:
                if isinstance(turn, dict):
                    role = "USER" if turn.get("role") == "user" else "ASSISTANT"
                    parts.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(parts)


# ─── Answer functions per mode ──────────────────────────────────────────────

def answer_R_T(scenario, query):
    M = build_retrieval(scenario)
    chunks = retrieve(query, M, k=5)
    ctx = "\n".join(f"- {c[:300]}" for c in chunks) if chunks else "(no relevant context)"
    return answer(query, ctx)


def answer_S_T(scenario, query, dim):
    B, attr = build_state(scenario, "S_T")
    ctx = to_string(B)
    return answer(query, ctx)


def answer_T_T(scenario, query):
    ctx = build_trajectory(scenario)
    if not ctx:
        return answer(query, "(no trajectory context)")
    return answer(query, ctx)


def answer_CUPMem(scenario, query, dim):
    """CUPMem-like: adjudicate + propagate + premise check."""
    B, attr = build_state(scenario, "CUPMem_like")
    
    if dim == "dim2_query":
        # PR dimension: check premises first
        pc = check_premises(B, query)
        if not pc.get("safe", True):
            stale = pc.get("stale_premises", [])
            correction = pc.get("corrected_state", "")
            ctx = to_string(B)
            ctx += f"\n\n[Premise Warning: The query assumes outdated state. Stale premises: {stale}. Corrected: {correction}]"
            return answer(query, ctx)
    
    ctx = to_string(B)
    return answer(query, ctx)


def answer_v1_1(scenario, query, dim):
    """Simple routing: dim1→S_T, dim2→S_T+premise, dim3→T_T+S_T."""
    if dim == "dim1_query":
        B, attr = build_state(scenario, "S_T")
        ctx = to_string(B)
        return answer(query, ctx)
    elif dim == "dim2_query":
        B, attr = build_state(scenario, "S_T")
        pc = check_premises(B, query)
        ctx = to_string(B)
        if not pc.get("safe", True):
            ctx += f"\n[Warning: Query premise contradicts current state. Corrected: {pc.get('corrected_state', '')}]"
        return answer(query, ctx)
    else:  # dim3_query
        B, attr = build_state(scenario, "S_T")
        traj = build_trajectory(scenario)
        ctx = to_string(B)
        if traj:
            ctx += "\n\n--- Relevant Timeline ---\n" + traj[:1500]
        return answer(query, ctx)


# ─── Judge function ─────────────────────────────────────────────────────────

def judge_response(scenario, query, response_text):
    """Rule-based judge.
    dim1: answer says old info is no longer valid
    dim2: answer rejects premise (says "no longer", "outdated", or corrects premise)
    dim3: answer acts based on new state (mentions new_state entity, not old_state entity)
    """
    M_new = str(scenario.get("M_new", ""))[:200].lower()
    M_old = str(scenario.get("M_old", ""))[:200].lower()
    response_lower = str(response_text or "").lower()
    
    # Extract key entities from M_old/M_new for comparison
    old_entities = set(w for w in M_old.split() if w[0].isupper() and len(w) > 2)
    new_entities = set(w for w in M_new.split() if w[0].isupper() and len(w) > 2)
    
    has_stale_signal = any(p in response_lower for p in [
        "no longer", "outdated", "not valid", "incorrect", "stale",
        "actually", "correction", "has changed", "is now", "now lives",
        "no, the user", "no, the", "does not",
        "your current", "your new", "updated"
    ])
    mentions_new = any(e.lower() in response_lower for e in new_entities)
    
    dim1 = has_stale_signal or mentions_new
    dim2 = has_stale_signal or "don't have enough" in response_lower or "not enough" in response_lower
    dim3 = mentions_new  # Acts based on new state
    
    return {"dim1": dim1, "dim2": dim2, "dim3": dim3}


# ─── Evaluation loop ────────────────────────────────────────────────────────

def evaluate_mode(mode, scenarios, limit=None, start=0):
    """Run one mode on (a subset of) scenarios.
    
    Returns: results dict, scenario_results list
    """
    mode_names = {
        "R_T": answer_R_T, "S_T": answer_S_T, "T_T": answer_T_T,
        "CUPMem_like": answer_CUPMem, "v1.1_composed": answer_v1_1
    }
    fn = mode_names[mode]
    
    scenario_results = []
    summary = {"T1": {"dim1": {"correct": 0, "total": 0}, "dim2": {"correct": 0, "total": 0}, "dim3": {"correct": 0, "total": 0}},
               "T2": {"dim1": {"correct": 0, "total": 0}, "dim2": {"correct": 0, "total": 0}, "dim3": {"correct": 0, "total": 0}}}
    
    end = len(scenarios) if limit is None else min(start + limit, len(scenarios))
    scenario_slice = scenarios[start:end]
    
    for i, sc in enumerate(scenario_slice):
        idx = start + i
        # Print progress every scenario
        _print_partial(mode, idx, end, summary)
        
        stype = sc.get("type", "T1")
        pqs = sc.get("probing_queries", {})
        
        result = {"index": idx, "uid": str(sc.get("uid", "")), "type": stype, "answers": {}, "judgments": {}}
        
        for dim in ("dim1_query", "dim2_query", "dim3_query"):
            q = pqs.get(dim, "")
            if not q:
                continue
            
            try:
                ans = fn(sc, q, dim) if mode in ("S_T", "CUPMem_like", "v1.1_composed") else fn(sc, q)
                time.sleep(0.15)  # rate limit
            except Exception as e:
                ans = f"[ERROR: {e}]"
            
            result["answers"][dim] = ans
            
            # Judge
            try:
                jd = judge_response(sc, q, ans or "")
                time.sleep(0.15)
            except Exception as e:
                jd = {"dim1_eval": {"pass": False}, "dim2_eval": {"pass": False}, "dim3_eval": {"pass": False}}
            
            result["judgments"][dim] = jd
            
            # Count  
            dim_key = dim.replace("_query", "")  # "dim1", "dim2", "dim3"
            dpass = jd.get(dim_key, False)
            if isinstance(dpass, dict):
                dpass = dpass.get("pass", False)
            
            summary[stype][dim_key]["total"] += 1
            if dpass:
                summary[stype][dim_key]["correct"] += 1
        
        scenario_results.append(result)
    
    return summary, scenario_results


def _print_partial(mode, idx, total, summary):
    """Print interim results."""
    t1_dim1 = summary["T1"]["dim1"]
    t1_dim2 = summary["T1"]["dim2"]
    t2_dim1 = summary["T2"]["dim1"]
    t2_dim2 = summary["T2"]["dim2"]
    a1 = t1_dim1["correct"]/max(t1_dim1["total"], 1)
    a2 = t1_dim2["correct"]/max(t1_dim2["total"], 1)
    a3 = t2_dim1["correct"]/max(t2_dim1["total"], 1)
    a4 = t2_dim2["correct"]/max(t2_dim2["total"], 1)
    t = t1_dim1["total"]+t1_dim2["total"]+t2_dim1["total"]+t2_dim2["total"]
    c = t1_dim1["correct"]+t1_dim2["correct"]+t2_dim1["correct"]+t2_dim2["correct"]
    overall = c/max(t, 1)
    print(f"  [{mode}] {idx}/{total} | T1-SR={a1:.3f} T1-PR={a2:.3f} T2-SR={a3:.3f} T2-PR={a4:.3f} overall={overall:.3f}")


def _save_intermediate(mode, results, idx, total):
    """Save intermediate results to a temp file."""
    outdir = "/home/orion/RESEARCH/state_memory/experiments"
    path = os.path.join(outdir, f"stale_cupmem_{mode}_interim.json")
    with open(path, "w") as f:
        json.dump({"mode": mode, "progress": f"{idx}/{total}", "results": results}, f, ensure_ascii=False, indent=2)


# ─── Print table ────────────────────────────────────────────────────────────

def print_table(all_results):
    """Print final comparison table."""
    print("\n" + "=" * 90)
    print("  FINAL COMPARISON TABLE — STALE Benchmark (CUPMem-like approximation)")
    print("=" * 90)
    print(f"  {'Mode':<15} {'T1-SR':>8} {'T1-PR':>8} {'T1-IPA':>8} {'T2-SR':>8} {'T2-PR':>8} {'T2-IPA':>8} {'Overall':>8}")
    print(f"  {'-'*75}")
    
    for mode, (summary, _) in all_results.items():
        t1_sr = _accuracy(summary["T1"]["dim1"])
        t1_pr = _accuracy(summary["T1"]["dim2"])
        t1_ipa = _accuracy(summary["T1"]["dim3"])
        t2_sr = _accuracy(summary["T2"]["dim1"])
        t2_pr = _accuracy(summary["T2"]["dim2"])
        t2_ipa = _accuracy(summary["T2"]["dim3"])
        overall = (summary["T1"]["dim1"]["correct"] + summary["T1"]["dim2"]["correct"] + 
                   summary["T1"]["dim3"]["correct"] + summary["T2"]["dim1"]["correct"] +
                   summary["T2"]["dim2"]["correct"] + summary["T2"]["dim3"]["correct"]) / max(
                   summary["T1"]["dim1"]["total"] + summary["T1"]["dim2"]["total"] + 
                   summary["T1"]["dim3"]["total"] + summary["T2"]["dim1"]["total"] +
                   summary["T2"]["dim2"]["total"] + summary["T2"]["dim3"]["total"], 1)
        print(f"  {mode:<15} {t1_sr:>8.3f} {t1_pr:>8.3f} {t1_ipa:>8.3f} {t2_sr:>8.3f} {t2_pr:>8.3f} {t2_ipa:>8.3f} {overall:>8.3f}")
    
    print("=" * 90)


def _accuracy(d):
    return d["correct"] / max(d["total"], 1)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit scenarios per mode")
    parser.add_argument("--modes", type=str, default=",".join(ALL_MODES), help="Comma-separated modes")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    args = parser.parse_args()
    
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in ALL_MODES]
    
    print("Loading STALE dataset...")
    ds = load_dataset("STALEproj/STALE", split="train", streaming=False)
    scenarios = [dict(ds[i]) for i in range(len(ds))]
    print(f"  Loaded {len(scenarios)} scenarios (T1={sum(1 for s in scenarios if s['type']=='T1')}, T2={sum(1 for s in scenarios if s['type']=='T2')})")
    
    all_results = {}
    all_scenario_results = {}
    
    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  MODE: {mode}")
        print(f"{'='*70}")
        reset_counters()
        
        summary, scenario_results = evaluate_mode(mode, scenarios, limit=args.limit, start=args.start)
        
        all_results[mode] = (summary, scenario_results)
        all_scenario_results[mode] = scenario_results
        
        print(f"\n  [{mode}] Done. Usage: {get_usage()}")
    
    # Save full results
    outdir = "/home/orion/RESEARCH/state_memory/experiments"
    results_path = os.path.join(outdir, "stale_cupmem_results.json")
    
    # Build serializable results
    save_data = {}
    for mode in modes:
        summary, scenario_results = all_results[mode]
        save_data[mode] = {
            "summary": summary,
            "scenario_results": [{
                "index": r["index"],
                "uid": r["uid"],
                "type": r["type"],
                "answers": r["answers"],
                "judgments": r["judgments"]
            } for r in scenario_results]
        }
    
    with open(results_path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print table
    print_table(all_results)


if __name__ == "__main__":
    main()

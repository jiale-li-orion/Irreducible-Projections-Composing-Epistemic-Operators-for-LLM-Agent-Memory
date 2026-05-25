#!/usr/bin/env python3
"""STALE full 400-scenario run: Optimal Phase-Routed System.

📄 Paper mapping: §6.2 External Generalization (Table 6) · Appendix F.1
dim1→S_T, dim2→S_T+premise, dim3→T_T, T2→T_T
"""

import sys, time, json, os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import load_dataset
from experiments.lib.llm_adapter import answer, get_usage, reset_counters
from experiments.lib.state import create_memory, add_belief, to_string
from experiments.lib.revision import revise
from experiments.lib.premise_verifier import check_premises

_OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_OUT_DIR, exist_ok=True)
OUT = os.path.join(_OUT_DIR, "stale_optimal_run.json")
LOG = os.path.join(os.path.dirname(__file__), "stale_optimal_run.log")

t_start = time.time()

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.time()-t_start:.0f}s] {msg}\n")
    print(msg, flush=True)

def judge(r):
    rl = (r or "").lower()
    return any(p in rl for p in [
        "no longer","outdated","not valid","is now","now lives","moved to",
        "has moved","does not","actually","your current","updated","incorrect",
        "don't have enough","not enough info"])

log("Loading STALE dataset...")
ds = load_dataset("STALEproj/STALE", split="train", streaming=False)
scenes = [dict(ds[i]) for i in range(len(ds))]
log(f"Loaded {len(scenes)} scenarios (T1={sum(1 for s in scenes if s['type']=='T1')}, T2={sum(1 for s in scenes if s['type']=='T2')})")

reset_counters()
per_dim = {"dim1":{"c":0,"n":0},"dim2":{"c":0,"n":0},"dim3":{"c":0,"n":0}}
breakdown = {"T1":{"dim1":{"c":0,"n":0},"dim2":{"c":0,"n":0},"dim3":{"c":0,"n":0}},
             "T2":{"dim1":{"c":0,"n":0},"dim2":{"c":0,"n":0},"dim3":{"c":0,"n":0}}}

results = []

for si, sc in enumerate(scenes):
    st = sc.get("type","T1")
    r_idx = sc.get("relevant_session_index",[])
    sessions = sc.get("haystack_session",[])
    tl = sc.get("timestamps",[])
    
    # Build state (for dim1/dim2)
    B = create_memory()
    for idx in range(len(sessions)):
        if idx in r_idx:
            i = r_idx.index(idx)
            ts = tl[idx] if idx<len(tl) and tl[idx] else f"s{idx}"
            if i == 0 and sc.get("M_old"):
                add_belief(B, "loc", str(sc["M_old"])[:200], ts, "user", 0.8)
            elif i == len(r_idx)-1 and sc.get("M_new"):
                revise(B, {"key":"loc","value":str(sc["M_new"])[:200],"timestamp":ts,"source":"user","confidence":0.9})
    
    # Build trajectory (for dim3 / T2)
    traj_parts = []
    for idx in sorted(r_idx):
        if idx < len(sessions):
            ts = tl[idx] if idx<len(tl) and tl[idx] else ""
            traj_parts.append(f"--- Session {idx} [{ts}] ---")
            for t in sessions[idx]:
                if isinstance(t, dict): traj_parts.append(f"{t.get('role','')}: {t.get('content','')}")
    traj = "\n".join(traj_parts) if traj_parts else "(empty)"
    
    answers = {}
    
    for d in ["dim1","dim2","dim3"]:
        q = sc["probing_queries"].get(d+"_query","")
        if not q:
            answers[d] = "(no query)"
            continue
        
        # Route
        if st == "T2" or d == "dim3":
            ctx = traj
        else:
            ctx = to_string(B)
            if d == "dim2":
                try:
                    pc = check_premises(B, q)
                    if not pc.get("safe", True):
                        ctx += f"\n[WARNING: outdated premise! Correction: {pc.get('corrected_state','')}]"
                except Exception as e:
                    ctx += "\n[WARNING: premise check failed]"
        
        try:
            ans = answer(q, ctx)
        except Exception as e:
            ans = f"[ERROR: {e}]"
        
        ok = judge(ans or "")
        per_dim[d]["c"] += 1 if ok else 0
        per_dim[d]["n"] += 1
        breakdown[st][d]["c"] += 1 if ok else 0
        breakdown[st][d]["n"] += 1
        answers[d] = ans
        time.sleep(0.06)
    
    results.append({
        "uid": str(sc.get("uid","")),
        "type": st,
        "answers": answers,
        "judged": {d: judge(answers.get(d,"")) for d in ["dim1","dim2","dim3"]}
    })
    
    if (si+1) % 25 == 0:
        d1 = per_dim["dim1"]
        d2 = per_dim["dim2"]
        d3 = per_dim["dim3"]
        ov = (d1["c"]+d2["c"]+d3["c"])/max(d1["n"]+d2["n"]+d3["n"],1)
        log(f"  {si+1}/{len(scenes)} | SR={d1['c']}/{d1['n']}={d1['c']/max(d1['n'],1):.3f} PR={d2['c']}/{d2['n']}={d2['c']/max(d2['n'],1):.3f} IPA={d3['c']}/{d3['n']}={d3['c']/max(d3['n'],1):.3f} Overall={ov:.3f}")

# Save
d1, d2, d3 = per_dim["dim1"], per_dim["dim2"], per_dim["dim3"]
sr = d1["c"]/max(d1["n"],1)
pr = d2["c"]/max(d2["n"],1)
ipa = d3["c"]/max(d3["n"],1)
ov = (d1["c"]+d2["c"]+d3["c"])/max(d1["n"]+d2["n"]+d3["n"],1)

t1_sr = breakdown["T1"]["dim1"]["c"]/max(breakdown["T1"]["dim1"]["n"],1)
t1_pr = breakdown["T1"]["dim2"]["c"]/max(breakdown["T1"]["dim2"]["n"],1)
t1_ipa = breakdown["T1"]["dim3"]["c"]/max(breakdown["T1"]["dim3"]["n"],1)
t2_sr = breakdown["T2"]["dim1"]["c"]/max(breakdown["T2"]["dim1"]["n"],1)
t2_pr = breakdown["T2"]["dim2"]["c"]/max(breakdown["T2"]["dim2"]["n"],1)
t2_ipa = breakdown["T2"]["dim3"]["c"]/max(breakdown["T2"]["dim3"]["n"],1)

output = {
    "configuration": "Optimal Phase-Routed: T1→S_T(dim1/dim2), T_T(dim3); T2→T_T(all)",
    "total_scenarios": len(scenes),
    "results": {
        "SR": {"correct": d1["c"], "total": d1["n"], "accuracy": sr},
        "PR": {"correct": d2["c"], "total": d2["n"], "accuracy": pr},
        "IPA": {"correct": d3["c"], "total": d3["n"], "accuracy": ipa},
        "Overall": (d1["c"]+d2["c"]+d3["c"])/(d1["n"]+d2["n"]+d3["n"]),
    },
    "breakdown": {
        "T1": {
            "SR": {"correct": breakdown["T1"]["dim1"]["c"], "total": breakdown["T1"]["dim1"]["n"], "accuracy": t1_sr},
            "PR": {"correct": breakdown["T1"]["dim2"]["c"], "total": breakdown["T1"]["dim2"]["n"], "accuracy": t1_pr},
            "IPA": {"correct": breakdown["T1"]["dim3"]["c"], "total": breakdown["T1"]["dim3"]["n"], "accuracy": t1_ipa},
        },
        "T2": {
            "SR": {"correct": breakdown["T2"]["dim1"]["c"], "total": breakdown["T2"]["dim1"]["n"], "accuracy": t2_sr},
            "PR": {"correct": breakdown["T2"]["dim2"]["c"], "total": breakdown["T2"]["dim2"]["n"], "accuracy": t2_pr},
            "IPA": {"correct": breakdown["T2"]["dim3"]["c"], "total": breakdown["T2"]["dim3"]["n"], "accuracy": t2_ipa},
        },
    },
    "api_usage": get_usage(),
    "elapsed_seconds": time.time() - t_start,
}

with open(OUT, "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

log("\n" + "="*60)
log("STALE FULL 400 — OPTIMAL PHASE-ROUTED SYSTEM")
log("="*60)
log(f"  {'':20} {'SR':>8} {'PR':>8} {'IPA':>8}")
log(f"  {'T1':20} {t1_sr:>8.3f} {t1_pr:>8.3f} {t1_ipa:>8.3f}")
log(f"  {'T2':20} {t2_sr:>8.3f} {t2_pr:>8.3f} {t2_ipa:>8.3f}")
log(f"  {'All':20} {sr:>8.3f} {pr:>8.3f} {ipa:>8.3f}")
log(f"  {'Overall':20} {ov:>8.3f}")
log(f"  CUPMem (paper, GPT-4o-mini): 0.680")
log(f"  Judge: rule-based (keyword) — may overcount")
log(f"  API calls: {get_usage().get('calls', 0)}, tokens: {get_usage().get('total_tokens', 0)}")
log(f"  Time: {time.time()-t_start:.0f}s")
log("Results saved to " + OUT)

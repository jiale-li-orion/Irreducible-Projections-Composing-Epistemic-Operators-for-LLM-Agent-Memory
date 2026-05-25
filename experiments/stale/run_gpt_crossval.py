#!/usr/bin/env python3
"""GPT-4o-mini on STALE 400 scenarios. T_T trajectory walk for all dimensions.
Routing: dim1→T_T, dim2→T_T, dim3→T_T, T2→T_T (GPT-4o-mini's dominant operator).
Judge: rule-based keyword (same as DS Flash STALE experiment).

Usage: setsid python3 stale_gpt.py > stale_gpt.log 2>&1 &
"""
import sys, json, time, os
sys.path.insert(0, "/home/orion/RESEARCH/state_memory")
from openai import OpenAI

OUT = "/home/orion/RESEARCH/state_memory/experiments/stale_gpt_results.json"
LOG_FILE = "/home/orion/RESEARCH/state_memory/experiments/stale_gpt.log"

def log(m):
    with open(LOG_FILE, "a") as f: f.write(f"{m}\n")
    print(m, flush=True)

from datasets import load_dataset
log("Loading STALE dataset...")
ds = load_dataset("STALEproj/STALE", split="train", streaming=False)
scenes = [dict(ds[i]) for i in range(400)]
log(f"Loaded {len(scenes)} scenarios (T1={sum(1 for s in scenes if s['type']=='T1')}, T2={sum(1 for s in scenes if s['type']=='T2')})")

client = OpenAI(api_key="sk-XVCpuWxgD53WTzK9TupYmgCHiIqNYjIoNQ5h10AdaLRyjpLq",
                base_url="https://api.echoflow.cn/v1", timeout=120)
ANS_SYS = "You are an agent answering questions based on your memory context."

def judge_keyword(r, sc):
    rl = (r or "").lower()
    stale = any(p in rl for p in ["no longer","outdated","not valid","is now","now lives","moved to",
        "has moved","does not","actually","your current","updated","incorrect","your premise","false premise",
        "don't have enough","not enough info"])
    return {"dim1": stale, "dim2": stale, "dim3": stale}

def build_traj(sc):
    """T_T: chronological walk of relevant sessions."""
    sessions = sc.get("haystack_session", [])
    relevant = sc.get("relevant_session_index", [])
    tl = sc.get("timestamps", [])
    parts = []
    for idx in sorted(relevant):
        if idx < len(sessions):
            ts = tl[idx] if idx < len(tl) and tl[idx] else ""
            parts.append(f"--- Session {idx} [{ts}] ---")
            for turn in sessions[idx]:
                if isinstance(turn, dict):
                    parts.append(f"{turn.get('role','')}: {turn.get('content','')}")
    return "\n".join(parts) if parts else "(empty)"

stats = {"T1": {"dim1":0,"dim2":0,"dim3":0,"n":0}, "T2": {"dim1":0,"dim2":0,"dim3":0,"n":0}}
results_list = []

for si, sc in enumerate(scenes):
    stype = sc.get("type", "T1")
    pqs = sc.get("probing_queries", {})
    traj = build_traj(sc)
    stats[stype]["n"] += 1
    answers = {}
    
    for dim_key, dim_name in [("dim1_query","dim1"), ("dim2_query","dim2"), ("dim3_query","dim3")]:
        q = pqs.get(dim_key, "")
        if not q:
            continue
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":ANS_SYS},
                          {"role":"user","content":f"Context:\n{traj}\n\nQuestion: {q}\n\nAnswer:"}],
                temperature=0.0, max_tokens=128)
            ans = resp.choices[0].message.content
        except Exception as e:
            ans = ""
        time.sleep(0.15)
        
        j = judge_keyword(ans or "", sc)
        if j.get(dim_name, False):
            stats[stype][dim_name] += 1
        answers[dim_key] = ans
    
    results_list.append({"uid": str(sc.get("uid","")), "type": stype, "answers": answers})
    
    if (si+1) % 50 == 0:
        t1 = stats["T1"]; t2 = stats["T2"]
        total_ok = t1["dim1"]+t1["dim2"]+t1["dim3"]+t2["dim1"]+t2["dim2"]+t2["dim3"]
        total_n = (t1["n"]+t2["n"])*3
        log(f"  {si+1}/400 | T1 SR={t1['dim1']/max(t1['n'],1):.3f} PR={t1['dim2']/max(t1['n'],1):.3f} IPA={t1['dim3']/max(t1['n'],1):.3f} | T2 SR={t2['dim1']/max(t2['n'],1):.3f} PR={t2['dim2']/max(t2['n'],1):.3f} IPA={t2['dim3']/max(t2['n'],1):.3f} | Overall={total_ok/max(total_n,1):.3f}")

# Final
t1 = stats["T1"]; t2 = stats["T2"]
total_ok = t1["dim1"]+t1["dim2"]+t1["dim3"]+t2["dim1"]+t2["dim2"]+t2["dim3"]
total_n = (t1["n"]+t2["n"])*3
overall = total_ok / max(total_n, 1)

log(f"\n{'='*60}")
log(f"GPT-4o-mini on STALE (T_T for all, rule judge)")
log(f"{'='*60}")
log(f"  {'':>10} {'SR':>8} {'PR':>8} {'IPA':>8}")
log(f"  {'T1':>10} {t1['dim1']/max(t1['n'],1):>8.3f} {t1['dim2']/max(t1['n'],1):>8.3f} {t1['dim3']/max(t1['n'],1):>8.3f}")
log(f"  {'T2':>10} {t2['dim1']/max(t2['n'],1):>8.3f} {t2['dim2']/max(t2['n'],1):>8.3f} {t2['dim3']/max(t2['n'],1):>8.3f}")
log(f"  {'Overall':>10} {'':>8} {'':>8} {overall:>8.3f}")
log(f"  CUPMem (GPT-4o-mini backbone): 0.680")
log(f"  Our DS Flash (DeepSeek Flash):  0.718")

out = {
    "model":"GPT-4o-mini","routing":"T_T for all","n_scenarios":400,
    "results": stats, "overall": overall,
    "per_scenario": results_list
}
with open(OUT, "w") as f: json.dump(out, f, ensure_ascii=False, indent=2)
log(f"\nSaved: {OUT}")

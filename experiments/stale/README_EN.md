# STALE Experiments [🇬🇧]

External validation on [STALE](https://arxiv.org/abs/2605.06527): operator composition routing on 400 conflict scenarios.

📄 **Paper mapping:**

| Experiment | Paper section |
|-----------|--------------|
| `run_composition.py` | §6.2 External Generalization (Table 6) · Appendix F.1 |

---

## Purpose

§6.2 asks: can the composition methodology transfer to an unseen external conflict benchmark (STALE) without redesign? Does operator complementarity hold in a conversational domain distinct from MSFT's entity-attribute structure?

## Configuration

| Parameter | Value |
|-----------|-------|
| Backbone LLM | DeepSeek Flash (`deepseek/deepseek-v4-flash`) |
| Temperature | 0.0 |
| Dataset | HuggingFace `STALEproj/STALE` (400 scenarios, 1200 queries) |
| Judge | Rule-based keyword (15 keywords) |
| Scenario types | T1 (direct conflict, 200) + T2 (propagation conflict, 200) |
| Probe dimensions | dim1 State Resolution, dim2 Premise Resistance, dim3 Implicit Policy Adaptation |

## Routing Strategy

| Type | dim1 | dim2 | dim3 |
|:----:|:----:|:----:|:----:|
| **T1** | S_T (state readout) | S_T + premise verifier | T_T (trajectory walk) |
| **T2** | T_T | T_T | T_T |

## Script Analysis

### run_composition.py

Runs all 400 scenarios in one pass (**no checkpoint** — restart from scratch if interrupted).

**Data loading (lines 39-42):**
```python
ds = load_dataset("STALEproj/STALE", split="train", streaming=False)
scenes = [dict(ds[i]) for i in range(len(ds))]
```
Loads STALE from HuggingFace (cached locally ~275MB). Each scenario contains:
- `M_old` / `M_new`: old/new state description text
- `haystack_session`: full dialogue history (list of sessions, each a list of messages)
- `relevant_session_index`: which sessions are relevant to this conflict
- `timestamps`: per-session timestamps
- `probing_queries`: three probe dimensions (`dim1_query`, `dim2_query`, `dim3_query`)
- `type`: `"T1"` (direct conflict) or `"T2"` (propagation conflict)

**State construction (lines 57-66):**
```python
B = create_memory()
for idx in range(len(sessions)):
    if idx in r_idx:
        i = r_idx.index(idx)
        if i == 0 and sc.get("M_old"):
            add_belief(B, "loc", str(sc["M_old"])[:200], ts, "user", 0.8)
        elif i == len(r_idx)-1 and sc.get("M_new"):
            revise(B, {"key":"loc","value":str(sc["M_new"])[:200], ...})
```
Only processes sessions in `relevant_session_index`. First relevant session → old belief (M_old, confidence=0.8). Last relevant session → new belief (M_new, confidence=0.9, triggers `revise` to overwrite old value). Other sessions are ignored.

**Trajectory construction (lines 68-76):**
```python
for idx in sorted(r_idx):
    traj_parts.append(f"--- Session {idx} [{ts}] ---")
    for t in sessions[idx]:
        traj_parts.append(f"{t.get('role','')}: {t.get('content','')}")
```
Chronologically sorts relevant sessions and concatenates into plain text. This is the T_T (trajectory operator) context.

**Routing logic (lines 80-97):**
```python
for d in ["dim1","dim2","dim3"]:
    if st == "T2" or d == "dim3":
        ctx = traj           # T2 all T_T, dim3 also T_T
    else:
        ctx = to_string(B)   # T1 dim1/dim2 use S_T
        if d == "dim2":
            pc = check_premises(B, q)  # premise verifier
            if not pc.get("safe", True):
                ctx += f"\n[WARNING: outdated premise! ...]"
```
- **T1 dim1** (State Resolution): S_T. Serializes state B to text, asks LLM for current state.
- **T1 dim2** (Premise Resistance): S_T + premise verifier. `check_premises` detects stale assumptions in the query; if found, injects an explicit warning into the context.
- **T1 dim3** (Implicit Policy Adaptation): T_T. Provides full dialogue trajectory for the LLM to infer updated-state actions.
- **T2 all dimensions**: T_T. State operator's write-time revision is ineffective for implicit propagation conflicts.

**Judge (lines 32-37):**
```python
def judge(r):
    rl = (r or "").lower()
    return any(p in rl for p in [
        "no longer","outdated","not valid","is now","now lives","moved to",
        "has moved","does not","actually","your current","updated","incorrect",
        "don't have enough","not enough info"])
```
Rule-based keyword judge (not LLM). Checks if the answer contains keywords indicating awareness of stale information:
- "Alice no longer lives in Seattle" → contains `"no longer"` → correct ✅
- "I don't have enough information" → contains `"don't have enough"` → correct ✅
- "Alice lives in Seattle" (no staleness awareness) → no keywords → wrong ❌

**Counting and output (lines 104-148):**
```python
per_dim[d]["c"] += 1 if ok else 0       # global
breakdown[st][d]["c"] += 1 if ok else 0  # per T1/T2
```
Progress printed every 25 scenarios. Final accuracy computed per dimension and per type, saved to `results/stale_optimal_run.json`.

---

## Results

| Type | SR | PR | IPA | Routing |
|:----:|:--:|:--:|:---:|---------|
| T1 (n=200) | 0.540 | **1.000** | 0.580 | S_T(dim1/2), T_T(dim3) |
| T2 (n=200) | 0.450 | 0.880 | 0.860 | T_T(all) |
| **All (n=400)** | **0.495** | **0.940** | **0.720** | **Overall: 0.718** |

## Data Provenance

Source: `results/stale_optimal_run.json`

| Paper data point | JSON path |
|-----------------|-----------|
| T1 SR/PR/IPA | `breakdown.T1.SR/PR/IPA.accuracy` |
| T2 SR/PR/IPA | `breakdown.T2.SR/PR/IPA.accuracy` |
| All SR/PR/IPA | `results.SR/PR/IPA.accuracy` |
| Overall | `results.Overall` |

---

## Reproducibility

### Environment

```bash
pip install -r ../../requirements.txt
export DEEPSEEK_API_KEY="sk-..."  # https://platform.deepseek.com
```

Python 3.12+. Requires `datasets` for loading STALE from HuggingFace.

### Data

Dataset loads automatically from HuggingFace: `STALEproj/STALE`.

### Run

```bash
python experiments/stale/run_composition.py
```

~22 minutes, ~1400 API calls. **No checkpoint resume** — restart from scratch if interrupted.

### Constraints

- Uses **rule-based keyword judge**, not LLM-as-judge. Keyword list in the script's `judge()` function
- T2 (propagation conflict) uses T_T for all dimensions, since the state operator cannot handle implicit propagation
- S_T + premise verifier achieves 1.000 on T1 dim2 but drops to 0.880 on T2
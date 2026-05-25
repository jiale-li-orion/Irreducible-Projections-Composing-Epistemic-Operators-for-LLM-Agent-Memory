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

Core experiment, runs all 400 scenarios in one pass.

**Data loading:**
- Loads 400 scenarios from HuggingFace `STALEproj/STALE`
- Each scenario: M_old (old state), M_new (new state), haystack_session (dialogue history), relevant_session_index, probing_queries (dim1/2/3)

**Core loop:**
1. Build state B from M_old/M_new (key-value)
2. Build trajectory traj (chronologically sorted relevant sessions)
3. Route per strategy:
   - dim1 → S_T (`to_string(B)`)
   - dim2 → S_T + premise verifier (`check_premises(B, q)`)
   - dim3 → T_T (`traj`)
   - T2 → all T_T
4. Rule-based judge: checks for keywords like "no longer", "outdated", "is now", etc.
5. Output: per-dimension correct/total, T1/T2 breakdown

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
# ATM-Bench Experiments

Oracle trajectory experiments on [ATM-Bench](https://arxiv.org/abs/2603.01990): multimodal long-term memory QA.

> **What question is this experiment series answering?**
>
> Do different memory access mechanisms (retrieval, state lookup, trajectory walk) behave fundamentally differently on realistic multimodal memory data? Is the failure of retrieval on this benchmark structural, not just a matter of better embedding or retrieval quality?

📄 **Paper mapping:**

| Experiment | Paper section |
|-----------|--------------|
| `run_full1013.py` | §6.3 Scaling Limits (Table 3) · Appendix C.1/C.3 |
| `run_hard31.py` | Appendix C.2 · §6.3 (Hard-31 reference) |

---

## What This Series Discovered

### 1. Information Scaling Reversal

When evidence format was upgraded from short summaries to full detail text:

| System | Short evidence | Full detail | Effect |
|--------|:-------------:|:-----------:|--------|
| Retrieval (R_T) | 34% | 24% | **Harmed** — embedding dilution |
| Trajectory (T_T) | 42% | 52% | **Helped** — richer signal for extraction |

Richer evidence helps trajectory but hurts retrieval. This is not an engineering artifact — it is a structural consequence of how each system consumes information:

- **Retrieval** operates over similarity rankings: longer text → lower similarity per token → evidence falls out of top-5
- **Trajectory** operates over chronological adjacency: longer text → more specific details → better answer extraction

### 2. Trajectory Bypasses the Retrieval Bottleneck

When ground-truth evidence was **not** in the top-5 retrieval window:

| Condition | Accuracy |
|-----------|:--------:|
| R_T | 0.077 |
| S_T | 0.051 |
| **T_T** | **0.362** |

Trajectory recovers 36% accuracy on questions where retrieval cannot even find the relevant evidence — because it uses ground-truth evidence IDs directly rather than similarity search.

### 3. Trajectory Advantage Grows with Task Complexity

| Evidence items | R_T | T_T | Gap |
|:-------------:|:---:|:---:|:---:|
| 1 | 0.450 | 0.542 | +0.092 |
| 2 | 0.236 | 0.405 | +0.169 |
| 3 | 0.314 | 0.510 | +0.196 |
| 4+ | 0.180 | 0.500 | **+0.320** |

The more evidence a question requires, the larger trajectory's advantage over retrieval.

### 4. No Single Operator is Sufficient

| Coverage pattern | % of QA |
|----------------|:-------:|
| Answered by any operator | 57.7% |
| **Unanswered by all three** | **42.3%** |
| Trajectory-unique | 16.3% |
| Retrieval-unique | 3.4% |

42% of questions cannot be answered by any of the three access mechanisms — suggesting the epistemic space exceeds the current operator basis.

### 5. Hard-31: The Hardest Cases

On the ATM-Bench-Hard subset (31 most difficult QA):

| R_T | S_T | T_T |
|:---:|:---:|:---:|
| 0.032 | 0.065 | **0.194** |

All three operators collapse near floor level, confirming this subset tests capabilities beyond simple evidence access.

---

## Files

| File | Purpose |
|------|---------|
| `run_full1013.py` | Full 1013 QA — §6.3 / Appendix C, 3 conditions (FS / State / Oracle Trajectory) |
| `run_hard31.py` | Hard-31 subset — Appendix C.2, same 3 conditions |
| `results/full1013_results.json` | Pre-computed results for Full 1013 QA (§6.3 / Appendix C) |
| `results/hard31_results.json` | Pre-computed results for Hard-31 (Appendix C.2) |

## Setup

```bash
# 1. Install dependencies
pip install -r ../../requirements.txt

# 2. Set API key (get one from https://platform.deepseek.com)
export DEEPSEEK_API_KEY="sk-..."
```

Requires Python 3.12+.

## Data

Download the ATM-Bench dataset from HuggingFace:

```bash
git clone https://huggingface.co/datasets/Jingbiao/ATM-Bench /path/to/data/ATM-Bench
```

Then either:

- Set `ATM_BENCH_DATA` environment variable: `export ATM_BENCH_DATA=/path/to/data/ATM-Bench/atm_data`
- Or place the dataset at `arxiv-submission/data/ATM-Bench/atm_data/` (default lookup path)

Expected data layout:

```
ATM-Bench/atm_data/data/
├── raw_memory/email/emails.json
├── processed_memory/image_batch_results.json
├── processed_memory/video_batch_results.json
└── atm-bench/atm-bench.json          (1013 QA)
└── atm-bench/atm-bench-hard.json     (31 QA, hard subset)
```

## Run

```bash
# Full 1013 QA (expect ~3 hours, ~5000 API calls)
python experiments/atm_bench/run_full1013.py

# Hard 31 (expect ~15 minutes, ~186 API calls)
python experiments/atm_bench/run_hard31.py
```

Both scripts auto-save checkpoints every 50 (or 10) QA and resume if interrupted.

## Results Summary

| Experiment | R_T | S_T | T_T |
|-----------|:---:|:---:|:---:|
| Full 1013 | 0.399 | 0.175 | **0.518** |
| Hard 31 | 0.032 | 0.065 | **0.194** |

## Notes

- All runs use DeepSeek Flash (`deepseek/deepseek-v4-flash`) at temperature 0.0
- Judge is the same LLM with a binary CORRECT/INCORRECT prompt
- `T_T` uses oracle trajectory access (ground-truth evidence IDs + chronological ordering)
- `S_T` on this benchmark is a Retrieval+KV variant (unstructured evidence prevents true state projection)
- The full detail/caption format (`detail[:1000]` / `caption[:1000]`) was used for all results above

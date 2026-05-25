# ATM-Bench Experiments

Oracle trajectory experiments on [ATM-Bench](https://arxiv.org/abs/2603.01990): multimodal long-term memory QA.

📄 **Paper mapping:**

| Experiment | Paper section |
|-----------|--------------|
| `run_full1013.py` | §6.3 Scaling Limits (Table 3) · Appendix C.1/C.3 |
| `run_hard31.py` | Appendix C.2 · §6.3 (Hard-31 reference) |

---

## Local Analysis Record

These experiments were conducted in three phases (2026-05-21 ~ 05-22). The full audit trail is documented in `reports/E12_atm_oracle_trajectory_results.md` and `for_arxiv/LOG_ATM_BENCH.md`.

### Experimental Progression

| Phase | QA | Evidence format | Date |
|:-----:|:--:|-----------------|:----:|
| B.1 (pilot) | 50 | `short_summary` / `short_caption` | 05-21 |
| B.2 (detail) | 50 | `detail[:1000]` / `caption[:1000]` | 05-21 |
| **B.3 (full)** | **1013** | `detail[:1000]` / `caption[:1000]` | **05-22** |
| Hard-31 | 31 | `detail[:1000]` / `caption[:1000]` | 05-22 |

### Key Finding: Information Scaling Reversal

Upgrading from short summaries to full detail revealed opposing responses:

| System | Short (B.1) | Full detail (B.2) | Change |
|--------|:-----------:|:-----------------:|:------:|
| Retrieval | 34.0% | 24.0% | -10% (harmed) |
| State | 24.0% | 14.0% | -10% (harmed) |
| Trajectory | 42.0% | 52.0% | **+10%** (helped) |

Richer evidence helps trajectory but hurts retrieval. The mechanism:

- **Retrieval** operates over similarity rankings: longer text → lower similarity per token → evidence-in-top-5 drops from 76% to 60%
- **Trajectory** bypasses embedding ranking (uses ground-truth evidence IDs): longer text → more specific numerical and contextual details → LLM can extract exact answers

### Trajectory Bypasses Retrieval Bottleneck

When evidence was **not** in top-5 retrieval window:

| Condition | Accuracy |
|-----------|:--------:|
| R_T | 0.077 |
| S_T | 0.051 |
| **T_T** | **0.362** |

Trajectory maintains 36% accuracy on questions where retrieval cannot find the evidence — it accesses memory through chronological ordering rather than similarity ranking.

### Trajectory Advantage Scales with Evidence Count

| Evidence items | n | R_T | T_T | Gap |
|:-------------:|:--:|:---:|:---:|:---:|
| 1 | 764 | 0.450 | 0.542 | +0.092 |
| 2 | 148 | 0.236 | 0.405 | +0.169 |
| 3 | 51 | 0.314 | 0.510 | +0.196 |
| 4+ | 50 | 0.180 | 0.500 | **+0.320** |

The gap widens monotonically with evidence count — structural advantage of temporal ordering over similarity search for multi-evidence reasoning.

### Condition Overlap (1013 QA)

| Pattern | Count | % |
|---------|:----:|:--:|
| All 3 correct | 143 | 14.1% |
| T_T unique | 165 | **16.3%** |
| R_T unique | 34 | 3.4% |
| S_T unique | 6 | 0.6% |
| None correct | 429 | **42.3%** |

### Hard-31 Ceiling

On the Hard subset (31 QA), all operators approach floor:

| R_T | S_T | T_T |
|:---:|:---:|:---:|
| 0.032 | 0.065 | 0.194 |

Evidence-in-top-5 rate drops to 0.548 (vs 0.616 for full 1013).

### Data Provenance

All results in this directory come from the **Oracle experiment** (`run_full1013.py` / `run_hard31.py`), which uses ground-truth evidence IDs for trajectory access. The earlier SGM experiment (`atm_sgm_experiment.py`) was conducted but excluded from the paper due to methodological concerns.

### Known Reproducibility Constraints

- All runs use DeepSeek Flash at temperature 0.0 — model version changes may shift absolute numbers
- No random seed is set in the scripts (though operations are deterministic: argsort + temperature=0.0)
- The ATM-Bench dataset (HuggingFace `Jingbiao/ATM-Bench`) is required; not included in this repo

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

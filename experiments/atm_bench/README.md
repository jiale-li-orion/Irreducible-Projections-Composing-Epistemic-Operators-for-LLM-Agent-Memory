# ATM-Bench Experiments

Oracle trajectory experiments on [ATM-Bench](https://arxiv.org/abs/2603.01990): multimodal long-term memory QA.

📄 **Paper mapping:**

| Experiment | Paper section |
|-----------|--------------|
| `run_full1013.py` | §6.3 Scaling Limits (Table 3) · Appendix C.1/C.3 |
| `run_hard31.py` | Appendix C.2 · §6.3 (Hard-31 reference) |

The paper's narrative (§6.3, Appendix C): ATM-Bench provides 1013 QA over four years of personal multimodal memory. Operator-level analysis reveals distinct epistemic coverage regions. $\mathcal{T}_T$ uniquely covers 16.3%, $\mathcal{R}_T$ uniquely 3.4%, and 42.3% are uncovered by any operator — empirical evidence that the epistemic space exceeds the current three-operator projection basis. The trajectory advantage grows with task complexity: from +0.092 on single-evidence queries to +0.320 on queries requiring 4+ evidence items.

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

# ATM-Bench Experiments

Oracle trajectory experiments on [ATM-Bench](https://arxiv.org/abs/2603.01990): multimodal long-term memory QA.

## Files

| File | Purpose |
|------|---------|
| `run_oracle_b3.py` | B.3 — Full 1013 QA, 3 conditions (FS / State / Oracle Trajectory) |
| `run_hard31.py` | Hard-31 subset, same 3 conditions |
| `results/` | Pre-computed result JSON files |

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
# Option A: clone the official repo
git clone https://huggingface.co/datasets/Jingbiao/ATM-Bench /path/to/data/ATM-Bench

# Option B: use the official GitHub repo
git clone https://github.com/atm-bench/ATM-Bench /path/to/data/ATM-Bench
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
python experiments/atm_bench/run_oracle_b3.py

# Hard 31 (expect ~15 minutes, ~186 API calls)
python experiments/atm_bench/run_hard31.py
```

Both scripts auto-save checkpoints every 50 (or 10) QA and resume if interrupted.

## Results

| Experiment | R_T | S_T | T_T |
|-----------|:---:|:---:|:---:|
| Full 1013 | 0.399 | 0.175 | **0.518** |
| Hard 31 | 0.032 | 0.065 | **0.194** |

## Notes

- All runs use DeepSeek Flash (deepseek-v4-flash) at temperature 0.0
- Judge is the same LLM with a binary CORRECT/INCORRECT prompt
- `T_T` uses oracle trajectory access (ground-truth evidence IDs + chronological ordering)
- `S_T` on this benchmark is a Retrieval+KV variant (unstructured evidence prevents true state projection)

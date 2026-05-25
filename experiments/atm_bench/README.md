# ATM-Bench Experiments

Oracle trajectory experiments on [ATM-Bench](https://arxiv.org/abs/2603.01990): multimodal long-term memory QA.

📄 **Paper mapping:**

| Experiment | Paper section |
|-----------|--------------|
| `run_full1013.py` | §6.3 Scaling Limits (Table 3) · Appendix C.1/C.3 |
| `run_hard31.py` | Appendix C.2 · §6.3 (Hard-31 reference) |

---

## 实验记录与分析

本目录的两个脚本对应论文 §6.3 和附录 C。以下数据均从 `results/` 下的 JSON 文件聚合得出。

### 三算子总体精度（full1013_results.json）

| 算子 | 精度 |
|------|:----:|
| R_T（top-5 检索） | 0.399 |
| S_T（Retrieval+KV） | 0.175 |
| **T_T（Oracle Trajectory）** | **0.518** |
| 证据 top-5 命中率 | 0.616 |

### 证据 top-5 条件精度

当证据在/不在 top-5 检索窗口内时，表现差异显著：

| 条件 | n | R_T | S_T | T_T |
|------|:--:|:---:|:---:|:---:|
| 证据在 top-5 内 | 624 | 0.599 | 0.252 | 0.615 |
| 证据不在 top-5 内 | 389 | 0.077 | 0.051 | **0.362** |

T_T 在检索完全找不到证据时仍保持 36.2% 精度——它通过时间顺序而非相似度排名访问记忆。

### 轨迹优势随证据数量增长

| 证据数 | n | R_T | T_T | 差距 |
|:------:|:--:|:---:|:---:|:----:|
| 1 | 764 | 0.450 | 0.542 | +0.092 |
| 2 | 148 | 0.236 | 0.405 | +0.169 |
| 3 | 51 | 0.314 | 0.510 | +0.196 |
| 4+ | 50 | 0.180 | 0.500 | **+0.320** |

差距随证据数量单调递增。

### 条件重叠（1013 QA）

| 模式 | 数量 | 占比 |
|------|:----:|:----:|
| 三个算子均正确 | 143 | 14.1% |
| **仅 T_T 正确** | **165** | **16.3%** |
| 仅 R_T 正确 | 34 | 3.4% |
| 仅 S_T 正确 | 6 | 0.6% |
| **全部错误** | **429** | **42.3%** |

### Hard-31（hard31_results.json）

| 算子 | R_T | S_T | T_T |
|:----:|:---:|:---:|:---:|
| 精度 | 0.032 | 0.065 | 0.194 |
| 证据 top-5 命中率 | 0.548 | | |

### 数据来源说明

所有结果来自 Oracle 实验（`run_full1013.py`、`run_hard31.py`），即使用 ground-truth 证据 ID 进行轨迹访问。S_T 在此基准上为 Retrieval+KV 变体（非结构化证据不支持真状态投影）。

### 可复现约束

- 全部运行使用 DeepSeek Flash，temperature=0.0
- 脚本未设置随机种子（运算过程确定：argsort + temperature=0.0）
- 需自行下载 ATM-Bench 数据集（HuggingFace `Jingbiao/ATM-Bench`），不包含在本仓库内

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

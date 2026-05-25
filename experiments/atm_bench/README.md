# ATM-Bench Experiments

Oracle trajectory experiments on [ATM-Bench](https://arxiv.org/abs/2603.01990): multimodal long-term memory QA.

📄 **Paper mapping:**

| Experiment | Paper section |
|-----------|--------------|
| `run_full1013.py` | §6.3 Scaling Limits (Table 3) · Appendix C.1/C.3 |
| `run_hard31.py` | Appendix C.2 · §6.3 (Hard-31 reference) |

---

## 实验记录与分析

实验分四个阶段进行（2026-05-21 ~ 05-22），完整审计追踪见 `reports/E12_atm_oracle_trajectory_results.md` 和 `for_arxiv/LOG_ATM_BENCH.md`。

### 实验演进

| 阶段 | QA 量 | 证据格式 | 日期 |
|:----:|:-----:|----------|:----:|
| B.1（先行） | 50 | `short_summary` / `short_caption` | 05-21 |
| B.2（细节） | 50 | `detail[:1000]` / `caption[:1000]` | 05-21 |
| **B.3（完整）** | **1013** | `detail[:1000]` / `caption[:1000]` | **05-22** |
| Hard-31 | 31 | `detail[:1000]` / `caption[:1000]` | 05-22 |

### 核心发现：信息规模的反向响应

从短摘要升级到全文细节后，三种算子的表现方向截然相反：

| 算子 | 短摘要 (B.1) | 全文细节 (B.2) | 变化 |
|------|:-----------:|:--------------:|:----:|
| 检索 R_T | 34.0% | 24.0% | -10%（受损） |
| 状态 S_T | 24.0% | 14.0% | -10%（受损） |
| **轨迹 T_T** | **42.0%** | **52.0%** | **+10%（获益）** |

更丰富的信息帮助了轨迹算子，却损害了检索算子。原因在于两者的信息消费机制不同：

- **检索算子**依靠相似度排序：文本越长 → 每 token 相似度越低 → 证据被挤出 top-5 窗口（命中率从 76% 降至 60%）
- **轨迹算子**绕过嵌入排序（直接使用 ground-truth 证据 ID）：文本越长 → 包含的具体数字和上下文越丰富 → LLM 越能提取出精确答案

### 轨迹算子绕过检索瓶颈

当证据不在 top-5 检索窗口内时：

| 条件 | 精度 |
|------|:----:|
| R_T | 0.077 |
| S_T | 0.051 |
| **T_T** | **0.362** |

轨迹在检索完全找不到证据的问题上仍保持 36% 的精度——因为它通过时间顺序而非相似度排名访问记忆。

### 轨迹优势随证据数量增长

| 证据数 | n | R_T | T_T | 差距 |
|:------:|:--:|:---:|:---:|:----:|
| 1 | 764 | 0.450 | 0.542 | +0.092 |
| 2 | 148 | 0.236 | 0.405 | +0.169 |
| 3 | 51 | 0.314 | 0.510 | +0.196 |
| 4+ | 50 | 0.180 | 0.500 | **+0.320** |

差距随证据数量单调递增——在需要多证据推理的问题上，时间排序的结构性优势远大于相似度搜索。

### 条件重叠（1013 QA）

| 模式 | 数量 | 占比 |
|------|:----:|:----:|
| 三个算子均正确 | 143 | 14.1% |
| **仅 T_T 正确** | **165** | **16.3%** |
| 仅 R_T 正确 | 34 | 3.4% |
| 仅 S_T 正确 | 6 | 0.6% |
| **全部错误** | **429** | **42.3%** |

### Hard-31 天花板

在 Hard 子集（31 个困难问题）上，所有算子接近下限：

| R_T | S_T | T_T |
|:---:|:---:|:---:|
| 0.032 | 0.065 | 0.194 |

证据 top-5 命中率从 1013 全集 0.616 降至 0.548。

### 数据来源说明

本目录所有结果来自 **Oracle 实验**（`run_full1013.py` / `run_hard31.py`），即使用 ground-truth 证据 ID 进行轨迹访问。此前还进行了 SGM 实验（`atm_sgm_experiment.py`），因方法论问题已被排除在论文之外。

### 可复现约束

- 全部运行使用 DeepSeek Flash，temperature=0.0——模型版本更新可能导致绝对数值偏移
- 脚本未设置随机种子（但运算过程确定：argsort + temperature=0.0）
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

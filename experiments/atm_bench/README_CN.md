# ATM-Bench Experiments [🇨🇳]

[ATM-Bench](https://arxiv.org/abs/2603.01990) 多模态长期记忆 QA 的 oracle 轨迹实验。

📄 **论文映射：**

| 实验 | 论文位置 |
|------|---------|
| `run_full1013.py` | §6.3 Scaling Limits（Table 3）· 附录 C.1/C.3 |
| `run_hard31.py` | 附录 C.2 · §6.3（Hard-31 引用） |

---

## 实验目的

论文 §6.3 和附录 C 试图回答：当认知空间超出当前三算子投影基时（即在 ATM-Bench 的大规模真实数据上），三个算子的覆盖边界在哪里？T_T 的 oracle 上限有多高？哪些查询是所有算子都无法回答的？

## 实验配置

| 参数 | 值 |
|------|-----|
| 基座模型 | DeepSeek Flash（`deepseek/deepseek-v4-flash`） |
| Temperature | 0.0 |
| 嵌入模型 | all-MiniLM-L6-v2（sentence-transformers），384 维 |
| Top-K | 5 |
| 证据格式 | email 用 `detail[:1000]`，图片/视频用 `caption[:1000]` |
| 裁判 | 同一 LLM，二元 CORRECT/INCORRECT 判定 |
| 数据集 | HuggingFace `Jingbiao/ATM-Bench`，~4 年多模态个人记忆，1013 QA |

## 实验设计

三个条件对比，共享同一 LLM、同一证据集合、同一裁判：

1. **R_T（Flat Search）**：top-5 embedding chunks，原始文本格式。`[Email/Photo/Video {ts}] {detail/caption[:1000]}`
2. **S_T（State，Retrieval+KV）**：top-5 chunks 以 key-value 形式呈现。`evidence_id: text[:200]`。注意：ATM-Bench 的证据是非结构化的自由文本，无法做真正的写时状态投影，因此 S_T 在此退化为 Retrieval+KV 变体。
3. **T_T（Oracle Trajectory）**：使用 ground-truth evidence_ids，按时间排序，step 格式。`[Step {i}: {ts}]\n  {text}`。这是 oracle 上限——绕过检索直接给正确证据。

## 脚本分析

### run_full1013.py

全量 1013 QA 实验，支持断点续跑。

**数据加载：**
- `load_all()` 从三个 JSON 文件加载记忆：
  - `emails.json`（~4.9MB，6742 条 email，取 `detail[:1000]`）
  - `image_batch_results.json`（~26.9MB，3759 张图片，取 `caption[:1000]`）
  - `video_batch_results.json`（~1.7MB，533 个视频，取 `caption[:1000]`）
- QA 从 `atm-bench.json`（1013 条，含 qtype 字段）加载

**核心流程：**
1. 用 `SentenceTransformer` 对所有记忆 item 做 embedding（`normalize_embeddings=True`）
2. 对所有 QA 问题做 embedding
3. 逐条 QA：
   - 计算 query embedding 与所有 memory embedding 的点积，取 top-5
   - 三种条件分别调用 LLM（`answer()`）并记录答案
   - 调用 LLM-as-judge（`judge()`）判定正确性
   - 每 50 QA 保存一次 checkpoint
   - 完成后清理 checkpoint 文件

**输出：** `results/full1013_results.json`，含 1013 条详细记录（每条含问题、真值、三个答案、裁判结果、evidence_ids、evidence_in_top5 状态）。

### run_hard31.py

结构与 run_full1013.py 相同，但 QA 来自 `atm-bench-hard.json`（31 个困难问题），checkpoint 每 10 QA 保存一次。

---

## 实验记录与分析

以下数据均从 `results/` 下的 JSON 文件直接聚合。

### 三算子总体精度

| 算子 | 精度 |
|------|:----:|
| R_T | 0.399 |
| S_T | 0.175 |
| **T_T** | **0.518** |
| 证据 top-5 命中率 | 0.616 |

### 证据 top-5 条件精度

| 条件 | n | R_T | S_T | T_T |
|------|:--:|:---:|:---:|:---:|
| 证据在 top-5 内 | 624 | 0.599 | 0.252 | 0.615 |
| 证据不在 top-5 内 | 389 | 0.077 | 0.051 | **0.362** |

T_T 在检索完全找不到证据时仍保持 36.2% 精度——通过时间顺序而非相似度排名访问记忆。

### 轨迹优势随证据数量增长

| 证据数 | n | R_T | T_T | 差距 |
|:------:|:--:|:---:|:---:|:----:|
| 1 | 764 | 0.450 | 0.542 | +0.092 |
| 2 | 148 | 0.236 | 0.405 | +0.169 |
| 3 | 51 | 0.314 | 0.510 | +0.196 |
| 4+ | 50 | 0.180 | 0.500 | **+0.320** |

差距随证据数量单调递增。

### 条件重叠

| 模式 | 数量 | 占比 |
|------|:----:|:----:|
| 三个算子均正确 | 143 | 14.1% |
| **仅 T_T 正确** | **165** | **16.3%** |
| 仅 R_T 正确 | 34 | 3.4% |
| 仅 S_T 正确 | 6 | 0.6% |
| **全部错误** | **429** | **42.3%** |

### Hard-31

| 算子 | R_T | S_T | T_T |
|:----:|:---:|:---:|:---:|
| 精度 | 0.032 | 0.065 | 0.194 |
| 证据 top-5 命中率 | 0.548 | | |

所有算子接近下限，确认该子集超出了简单证据访问的能力范围。

---

## 可复现说明

### 环境

```bash
pip install -r ../../requirements.txt
export DEEPSEEK_API_KEY="sk-..."  # https://platform.deepseek.com
```

Python 3.12+。

### 数据

```bash
git clone https://huggingface.co/datasets/Jingbiao/ATM-Bench /path/to/data/ATM-Bench
export ATM_BENCH_DATA=/path/to/data/ATM-Bench/atm_data
```

数据集不包含在本仓库内。

### 运行

```bash
python experiments/atm_bench/run_full1013.py   # ~3 小时，~5000 API 调用
python experiments/atm_bench/run_hard31.py     # ~15 分钟，~186 API 调用
```

两脚本均支持 checkpoint 断点续跑。

### 约束

- 脚本未设置随机种子（运算确定：argsort + temperature=0.0）
- 模型版本更新可能导致绝对数值偏移。原实验使用即将废弃的 `deepseek-chat`，该名称将于 2026/07/24 后不再可用。当前 `deepseek-v4-flash` 默认启用推理模式（thinking mode），每次输出需消耗 reasoning_tokens。原设 `max_tokens=16`（judge）和 `128`（answer）已被调至 `256`。详见 `experiments/lib/llm_adapter.py`。
- 本目录文件名对应论文 §6.3 和附录 C。S_T 在此为 Retrieval+KV 变体，非真状态投影。T_T 使用 oracle 轨迹访问（ground-truth 证据 ID + 时间排序），代表认知上限而非可部署系统精度。

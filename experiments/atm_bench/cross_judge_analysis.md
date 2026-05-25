# Cross-Judge Validation: GPT-4o-mini vs DeepSeek Flash

> 实验日期: 2026-05-25
> 目的: 回应 §7 "Single judge model" limitation
> 脚本: `cross_judge_gpt4o.py`（本地 `state_memory/experiments/`）
> 结果: `results/cross_judge_gpt4o_full1013.json`, `results/cross_judge_gpt4o_hard31.json`

---

## 实验设计

- 使用 `full1013_results.json` 中已有的答案，**不重新生成**
- 用 GPT-4o-mini 作为 judge，prompt、temperature、message format 与原始 DeepSeek Flash judge **完全一致**
- 唯一变量：model + API endpoint
- 支持 checkpoint 断点续跑（每 50 QA 保存一次）

## 结果

### Full 1013

| 指标 | 值 |
|------|:---:|
| 总判定数 | 3039（1013 QA × 3 算子） |
| 一致数 | 2860 |
| **一致率** | **94.1%** |
| 不一致数 | 179 |

### Hard-31

| 指标 | 值 |
|------|:---:|
| 总判定数 | 93（31 QA × 3 算子） |
| 一致数 | 89 |
| **一致率** | **95.7%** |
| 不一致数 | 4 |

---

## 不一致分析

### 按算子

| 算子 | 不一致数 | 总判定 | 不一致率 |
|:----:|:--------:|:------:|:--------:|
| R_T | 55 | 1013 | 5.4% |
| S_T | 40 | 1013 | 3.9% |
| **T_T** | **84** | **1013** | **8.3%** |

T_T 不一致率最高，与 T_T 答案更长、含更多上下文细节有关（不同 judge 对"答案包含正确信息但附带了额外上下文"的判断标准不一）。

### 按方向

| 方向 | 数量 |
|------|:----:|
| DeepSeek=OK, GPT-4o=WRONG | **103** |
| DeepSeek=WRONG, GPT-4o=OK | **76** |

不对称幅度约 27/3039 = **0.9%**，无系统性偏倚。排除 abstention 相关分歧后，方向分布基本对称（47 vs 51）。

### 按题型

| 题型 | 不一致 / 总判定 | 不一致率 |
|:----:|:--------------:|:--------:|
| number | 15 / 1080 | 1.4% |
| open_end | 100 / 1542 | 6.5% |
| **list_recall** | **64 / 417** | **15.3%** |

List_recall 分歧率最高，因为列表类问题的部分匹配 vs 完全匹配边界模糊。

### 分歧根因分类

| 类型 | 数量 | 占比 |
|------|:----:|:----:|
| Abstention（"不确定"的回答是否算错） | 81 | 45.3% |
| 答案含额外上下文（LLM 补充了 GT 之外的信息） | ~30 | ~17% |
| 部分匹配（答对核心但遗漏细节） | ~30 | ~17% |
| 其他/不可分类 | ~38 | ~21% |

---

## 结论

1. **94.1% 跨模型一致率**——结果对 judge 模型选择不敏感
2. **无系统性方向偏倚**——两个模型互有松紧，基本对称
3. **分歧集中在结构性原因**（abstention 判断、长答案的边界判定），而非 judge 偏见
4. 可用于回应 §7 Limitations 中的 "Single judge model" 担忧
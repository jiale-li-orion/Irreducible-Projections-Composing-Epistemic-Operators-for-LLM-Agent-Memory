# STALE 实验 [🇨🇳]

External validation on [STALE](https://arxiv.org/abs/2605.06527): 400 冲突场景的算子组合路由实验。

📄 **论文映射：**

| 实验 | 论文位置 |
|------|---------|
| `run_composition.py` | §6.2 External Generalization（Table 6）· 附录 F.1 |

---

## 实验目的

论文 §6.2 试图回答：组合路由方法论能否**无需重新设计**地迁移到 STALE 这个外部冲突基准？在对话域（与 MSFT 的实体-属性结构不同）上，算子的互补性是否仍然成立？

## 实验配置

| 参数 | 值 |
|------|-----|
| 基座模型 | DeepSeek Flash（`deepseek/deepseek-v4-flash`） |
| Temperature | 0.0 |
| 数据集 | HuggingFace `STALEproj/STALE`（400 场景，1200 查询） |
| 裁判 | Rule-based keyword（15 个关键词） |
| 场景类型 | T1（直接冲突，200 场景）+ T2（传播冲突，200 场景） |
| 探测维度 | dim1 State Resolution、dim2 Premise Resistance、dim3 Implicit Policy Adaptation |

## 路由策略

| 类型 | dim1 | dim2 | dim3 |
|:----:|:----:|:----:|:----:|
| **T1** | S_T（状态读入） | S_T + premise verifier | T_T（轨迹遍历） |
| **T2** | T_T | T_T | T_T |

## 脚本分析

### run_composition.py

核心实验脚本，一次性跑完 400 场景。

**数据加载：**
- 从 HuggingFace `STALEproj/STALE` 加载 400 场景
- 每个场景含：M_old（旧状态）、M_new（新状态）、haystack_session（对话历史）、relevant_session_index、probing_queries（dim1/2/3）

**核心流程：**
1. 对每个场景构建状态 B（从 M_old/M_new 提取 key-value）
2. 构建轨迹 traj（按时间排序的相关会话）
3. 按路由策略分发：
   - dim1 用 S_T（`to_string(B)`）
   - dim2 用 S_T + premise verifier（`check_premises(B, q)`）
   - dim3 用 T_T（`traj`）
   - T2 全部用 T_T
4. Rule-based judge：检查回答中是否包含 "no longer"、"outdated"、"is now" 等关键词
5. 输出：每个维度正确/总数，按 T1/T2 分解

---

## 结果

| 类型 | SR | PR | IPA | 路由 |
|:----:|:--:|:--:|:---:|------|
| T1（n=200） | 0.540 | **1.000** | 0.580 | S_T(dim1/2), T_T(dim3) |
| T2（n=200） | 0.450 | 0.880 | 0.860 | T_T(all) |
| **All（n=400）** | **0.495** | **0.940** | **0.720** | **Overall: 0.718** |

## 数据溯源

来源：`results/stale_optimal_run.json`

| 论文数据点 | JSON 路径 |
|-----------|-----------|
| T1 SR/PR/IPA | `breakdown.T1.SR/PR/IPA.accuracy` |
| T2 SR/PR/IPA | `breakdown.T2.SR/PR/IPA.accuracy` |
| All SR/PR/IPA | `results.SR/PR/IPA.accuracy` |
| Overall | `results.Overall` |

---

## 可复现说明

### 环境

```bash
pip install -r ../../requirements.txt
export DEEPSEEK_API_KEY="sk-..."  # https://platform.deepseek.com
```

Python 3.12+，需额外安装 `datasets`（用于加载 STALE 数据集）。

### 数据

数据集从 HuggingFace 自动加载：`STALEproj/STALE`，无需手动下载。

### 运行

```bash
python experiments/stale/run_composition.py
```

约 22 分钟，1400 次 API 调用。**无 checkpoint 机制**，中断需重跑。

### 约束

- 使用 **rule-based keyword judge**，非 LLM judge。关键词列表见脚本 `judge()` 函数
- T2（传播冲突）全部用 T_T 处理，因状态算子在传播冲突上无效
- S_T + premise verifier 在 dim2 上达到 1.000（T1），但 T2 上降为 0.880
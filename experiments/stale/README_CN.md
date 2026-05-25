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

核心实验脚本，一次性跑完 400 场景（**无 checkpoint**，中断需重跑）。

**数据加载（第 39-42 行）：**
```python
ds = load_dataset("STALEproj/STALE", split="train", streaming=False)
scenes = [dict(ds[i]) for i in range(len(ds))]
```
从 HuggingFace 加载 STALE 数据集（400 条），已缓存到本地约 275MB。每个场景含：
- `M_old` / `M_new`：旧/新状态描述文本
- `haystack_session`：全部对话历史（session 列表，每个 session 是 message 列表）
- `relevant_session_index`：涉及当前冲突的 session 索引
- `timestamps`：每个 session 的时间戳
- `probing_queries`：三个探测维度的查询（`dim1_query`、`dim2_query`、`dim3_query`）
- `type`：`"T1"`（直接冲突）或 `"T2"`（传播冲突）

**状态构建（第 57-66 行）：**
```python
B = create_memory()
for idx in range(len(sessions)):
    if idx in r_idx:
        i = r_idx.index(idx)
        ts = tl[idx] if idx<len(tl) and tl[idx] else f"s{idx}"
        if i == 0 and sc.get("M_old"):
            add_belief(B, "loc", str(sc["M_old"])[:200], ts, "user", 0.8)
        elif i == len(r_idx)-1 and sc.get("M_new"):
            revise(B, {"key":"loc","value":str(sc["M_new"])[:200], ...})
```
只处理 `relevant_session_index` 列出的 session。第一个相关 session 写入旧信念（M_old，confidence=0.8），最后一个写入新信念（M_new，confidence=0.9，触发 `revise` 覆盖旧值）。非相关的 session 直接忽略。

**轨迹构建（第 68-76 行）：**
```python
traj_parts = []
for idx in sorted(r_idx):
    traj_parts.append(f"--- Session {idx} [{ts}] ---")
    for t in sessions[idx]:
        traj_parts.append(f"{t.get('role','')}: {t.get('content','')}")
```
将相关 session 按时间排序，拼接成纯文本格式。包含完整对话内容。这是 T_T（轨迹算子）的上下文。

**路由逻辑（第 80-97 行）：**
```python
for d in ["dim1","dim2","dim3"]:
    if st == "T2" or d == "dim3":
        ctx = traj           # T2 全部用 T_T, dim3 也用 T_T
    else:
        ctx = to_string(B)   # T1 dim1/dim2 用 S_T
        if d == "dim2":
            pc = check_premises(B, q)  # premise verifier
            if not pc.get("safe", True):
                ctx += f"\n[WARNING: outdated premise! ...]"
```
- **T1 dim1**（State Resolution）：S_T。直接将状态 B 序列化为文本，询问 LLM 当前状态是什么
- **T1 dim2**（Premise Resistance）：S_T + premise verifier。先用 `check_premises` 检测查询中是否嵌有陈旧假设，若有则在上下文中加入显式警告
- **T1 dim3**（Implicit Policy Adaptation）：T_T。提供完整对话轨迹，让 LLM 基于更新后的状态给出行动建议
- **T2 全部维度**：T_T。传播冲突中状态算子的写时修订无效（因为冲突是隐式的），全部交给轨迹算子处理

**Judge（第 32-37 行）：**
```python
def judge(r):
    rl = (r or "").lower()
    return any(p in rl for p in [
        "no longer","outdated","not valid","is now","now lives","moved to",
        "has moved","does not","actually","your current","updated","incorrect",
        "don't have enough","not enough info"])
```
Rule-based keyword judge，非 LLM 裁判。判断 LLM 的回答是否包含识别出陈旧信息的关键词。例如：
- "Alice no longer lives in Seattle" → 含 `"no longer"` → 判对 ✅
- "I don't have enough information" → 含 `"don't have enough"` → 判对 ✅
- "Alice lives in Seattle"（直接回答，未察觉陈旧）→ 无关键词 → 判错 ❌

**计数与输出（第 104-148 行）：**
```python
per_dim[d]["c"] += 1 if ok else 0  # 全局计数
breakdown[st][d]["c"] += 1 if ok else 0  # 按 T1/T2 分解
```
每 25 场景打印一次进度。结束时计算各维度精度，保存到 `results/stale_optimal_run.json`。

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
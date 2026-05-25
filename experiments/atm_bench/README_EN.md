# ATM-Bench Experiments [🇬🇧]

Oracle trajectory experiments on [ATM-Bench](https://arxiv.org/abs/2603.01990): multimodal long-term memory QA.

📄 **Paper mapping:**

| Experiment | Paper section |
|-----------|--------------|
| `run_full1013.py` | §6.3 Scaling Limits (Table 3) · Appendix C.1/C.3 |
| `run_hard31.py` | Appendix C.2 · §6.3 (Hard-31 reference) |

---

## Purpose

§6.3 and Appendix C ask: when the epistemic space exceeds the current three-operator projection basis (i.e., on ATM-Bench's large-scale realistic data), where are the coverage boundaries of the three operators? What is the T_T oracle upper bound? Which queries are answerable by none of the operators?

## Configuration

| Parameter | Value |
|-----------|-------|
| Backbone LLM | DeepSeek Flash (`deepseek/deepseek-v4-flash`) |
| Temperature | 0.0 |
| Embedding model | all-MiniLM-L6-v2 (sentence-transformers), 384-dim |
| Top-K | 5 |
| Evidence format | email: `detail[:1000]`, image/video: `caption[:1000]` |
| Judge | Same LLM, binary CORRECT/INCORRECT |
| Dataset | HuggingFace `Jingbiao/ATM-Bench`, ~4 years multimodal personal memory, 1013 QA |

## Design

Three conditions sharing the same LLM, same evidence pool, same judge:

1. **R_T (Flat Search)**: top-5 embedding chunks, raw text format. `[Email/Photo/Video {ts}] {detail/caption[:1000]}`
2. **S_T (State, Retrieval+KV)**: top-5 chunks as key-value pairs. `evidence_id: text[:200]`. Note: ATM-Bench evidence is unstructured free text, so S_T degrades to a Retrieval+KV variant — true write-time state projection is not possible.
3. **T_T (Oracle Trajectory)**: ground-truth evidence_ids, chronologically sorted, step format. `[Step {i}: {ts}]\n  {text}`. This is an oracle upper bound — it bypasses retrieval entirely.

## Script Analysis

### run_full1013.py

Full 1013 QA experiment with checkpoint resume.

**Data loading:**
- `load_all()` loads memory from three JSON files:
  - `emails.json` (~4.9MB, 6742 emails, `detail[:1000]`)
  - `image_batch_results.json` (~26.9MB, 3759 images, `caption[:1000]`)
  - `video_batch_results.json` (~1.7MB, 533 videos, `caption[:1000]`)
- QA loaded from `atm-bench.json` (1013 items with qtype field)

**Core loop:**
1. Embed all memory items with `SentenceTransformer` (`normalize_embeddings=True`)
2. Embed all QA questions
3. For each QA:
   - Compute dot product between query embedding and all memory embeddings, take top-5
   - Call LLM (`answer()`) for each of the three conditions
   - Call LLM-as-judge (`judge()`) for correctness
   - Save checkpoint every 50 QA
   - Clean up checkpoint on completion

**Output:** `results/full1013_results.json`, 1013 detailed records (question, ground truth, 3 answers, judge results, evidence_ids, evidence_in_top5 status).

### run_hard31.py

Same structure as run_full1013.py, QA sourced from `atm-bench-hard.json` (31 hard questions), checkpoint every 10 QA.

---

## Results

All data below is aggregated directly from JSON files in `results/`.

### Overall Operator Accuracy

| Operator | Accuracy |
|----------|:--------:|
| R_T | 0.399 |
| S_T | 0.175 |
| **T_T** | **0.518** |
| Evidence in top-5 rate | 0.616 |

### Evidence-in-Top-5 Conditional Accuracy

| Condition | n | R_T | S_T | T_T |
|-----------|:--:|:---:|:---:|:---:|
| Evidence in top-5 | 624 | 0.599 | 0.252 | 0.615 |
| NOT in top-5 | 389 | 0.077 | 0.051 | **0.362** |

T_T maintains 36.2% accuracy on questions where retrieval fails to surface the evidence — it accesses memory through chronological ordering rather than similarity ranking.

### Trajectory Advantage vs Evidence Count

| Evidence items | n | R_T | T_T | Gap |
|:-------------:|:--:|:---:|:---:|:---:|
| 1 | 764 | 0.450 | 0.542 | +0.092 |
| 2 | 148 | 0.236 | 0.405 | +0.169 |
| 3 | 51 | 0.314 | 0.510 | +0.196 |
| 4+ | 50 | 0.180 | 0.500 | **+0.320** |

Gap increases monotonically with evidence count.

### Condition Overlap

| Pattern | Count | % |
|---------|:----:|:--:|
| All 3 correct | 143 | 14.1% |
| **T_T only** | **165** | **16.3%** |
| R_T only | 34 | 3.4% |
| S_T only | 6 | 0.6% |
| **None correct** | **429** | **42.3%** |

### Hard-31

| Operator | R_T | S_T | T_T |
|:--------:|:---:|:---:|:---:|
| Accuracy | 0.032 | 0.065 | 0.194 |
| Evidence in top-5 | 0.548 | | |

All operators approach floor, confirming this subset tests capabilities beyond simple evidence access.

---

## Reproducibility

### Environment

```bash
pip install -r ../../requirements.txt
export DEEPSEEK_API_KEY="sk-..."  # https://platform.deepseek.com
```

Python 3.12+.

### Data

```bash
git clone https://huggingface.co/datasets/Jingbiao/ATM-Bench /path/to/data/ATM-Bench
export ATM_BENCH_DATA=/path/to/data/ATM-Bench/atm_data
```

The dataset is not included in this repository.

### Run

```bash
python experiments/atm_bench/run_full1013.py   # ~3 hours, ~5000 API calls
python experiments/atm_bench/run_hard31.py     # ~15 min, ~186 API calls
```

Both scripts support checkpoint resume.

### Constraints

- No random seed is set (operations are deterministic: argsort + temperature=0.0)
- Model version changes may shift absolute numbers. The original experiments used the legacy `deepseek-chat` name (to be deprecated 2026/07/24). The current `deepseek-v4-flash` enables thinking mode by default, consuming reasoning_tokens on every output. The original `max_tokens=16` (judge) and `128` (answer) have been raised to `256`. See `experiments/lib/llm_adapter.py`.
- Experiment files in this directory correspond to §6.3 and Appendix C. S_T on this benchmark is a Retrieval+KV variant rather than true state projection. T_T uses oracle trajectory access (ground-truth evidence IDs + chronological ordering), representing an epistemic upper bound rather than a deployable system accuracy.

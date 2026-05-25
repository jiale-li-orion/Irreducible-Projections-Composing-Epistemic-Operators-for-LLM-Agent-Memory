"""ATM-Bench oracle trajectory — Full 1013 QA (§6.3 / Appendix C).

Runs all 1013 QA through three conditions:
    1. FS (embedding chunks, baseline)
    2. State (key-value pairs, baseline)
    3. Oracle Trajectory (ground-truth evidence_ids, chronologically sorted)

Checkpoint resume: saves progress every 50 QA; resumes from checkpoint if interrupted.
"""
import os, sys, json, time, datetime, numpy as np
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # silence HF warning

# --- Repo-aware paths ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.lib.llm_adapter import answer, judge, get_usage, reset_counters, MODEL, ANSWER_SYSTEM_PROMPT
from sentence_transformers import SentenceTransformer

# Data: download from https://huggingface.co/datasets/Jingbiao/ATM-Bench
# Set ATM_BENCH_DATA env var to point to local copy; default expects it at ../data/
_DATA_HOME = os.environ.get("ATM_BENCH_DATA", os.path.join(REPO_ROOT, "data", "ATM-Bench", "atm_data"))
DATA_DIR = _DATA_HOME

K = 5
_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(_CHECKPOINT_DIR, "full1013_checkpoint.json")

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
RESULT_PATH = os.path.join(_RESULTS_DIR, "full1013_results.json")

def eid(p):
    return os.path.splitext(os.path.basename(p))[0]

def load_all():
    items = {}
    with open(os.path.join(DATA_DIR, "data", "raw_memory", "email", "emails.json")) as f:
        for em in json.load(f):
            items[em["id"]] = {
                "text": f"[Email {em['timestamp']}] {em['detail'][:1000]}",
                "timestamp": em["timestamp"],
            }
    with open(os.path.join(DATA_DIR, "data", "processed_memory", "image_batch_results.json")) as f:
        for img in json.load(f):
            e = eid(img.get("image_path", ""))
            if e:
                items[e] = {
                    "text": f"[Photo {img.get('timestamp','')}] {img.get('caption','')[:1000]}",
                    "timestamp": img.get("timestamp", ""),
                }
    with open(os.path.join(DATA_DIR, "data", "processed_memory", "video_batch_results.json")) as f:
        for vid in json.load(f):
            e = eid(vid.get("video_path", ""))
            if e:
                items[e] = {
                    "text": f"[Video {vid.get('timestamp','')}] {vid.get('caption','')[:1000]}",
                    "timestamp": vid.get("timestamp", ""),
                }
    return items

def build_trajectory(evidence_ids, memory):
    items = []
    for eid_ in evidence_ids:
        if eid_ in memory:
            items.append((eid_, memory[eid_]))
    items.sort(key=lambda x: x[1]["timestamp"])
    steps = []
    for step_idx, (eid_, item) in enumerate(items, 1):
        ts = item["timestamp"]
        text = item["text"]
        steps.append(f"[Step {step_idx}: {ts}]\n  {text}")
    return "\n\n".join(steps)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            cp = json.load(f)
        print(f"Resumed checkpoint: {cp['qa_completed']}/{cp['qa_total']} completed")
        return cp
    return None

def save_checkpoint(qa_completed, qa_total, details, fs_c, st_c, traj_c, top5_c, start_time):
    cp = {
        "qa_completed": qa_completed,
        "qa_total": qa_total,
        "fs_correct": fs_c,
        "state_correct": st_c,
        "trajectory_correct": traj_c,
        "evidence_in_top5_count": top5_c,
        "details": details,
        "elapsed_seconds": (datetime.datetime.utcnow() - start_time).total_seconds(),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(cp, f, indent=2)

def main():
    print("Loading data...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    memory = load_all()
    with open(os.path.join(DATA_DIR, "data", "atm-bench", "atm-bench.json")) as f:
        qa = json.load(f)
    print(f"Memory: {len(memory)} items, QA: {len(qa)}")

    print("\nEmbedding memory items...")
    mem_texts = [item["text"] for item in memory.values()]
    mem_ids = list(memory.keys())
    mem_embeds = model.encode(mem_texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"Memory embeddings done. Shape: {mem_embeds.shape}")

    print("Embedding QA questions...")
    q_embeds = model.encode([q["question"] for q in qa], normalize_embeddings=True, show_progress_bar=True)
    print(f"QA embeddings done. Shape: {q_embeds.shape}")

    N = 1013  # full ATM-Bench 1013 QA
    reset_counters()
    start_time = datetime.datetime.utcnow()

    # Load checkpoint state
    checkpoint = load_checkpoint()
    if checkpoint:
        details = checkpoint["details"]
        qa_completed = checkpoint["qa_completed"]
        fs_correct = checkpoint["fs_correct"]
        st_correct = checkpoint["state_correct"]
        traj_correct = checkpoint["trajectory_correct"]
        evidence_in_top5_count = checkpoint["evidence_in_top5_count"]
        start_idx = qa_completed  # resume from next uncompleted
        print(f"Resuming from index {start_idx}")
    else:
        details = []
        qa_completed = 0
        fs_correct, st_correct, traj_correct = 0, 0, 0
        evidence_in_top5_count = 0
        start_idx = 0

    print(f"\n{'='*60}")
    print(f"Evaluating {N} QA across 3 conditions (starting from index {start_idx})")
    print(f"{'='*60}")

    for i in range(start_idx, N):
        item = qa[i]
        q, gt = item["question"], item["answer"]
        evidence_ids = item["evidence_ids"]
        scores = mem_embeds @ q_embeds[i]
        top_idx = np.argsort(-scores)[:K]

        top5_ids = set(mem_ids[j] for j in top_idx)
        evidence_in_top5 = any(eid_ in top5_ids for eid_ in evidence_ids)
        if evidence_in_top5:
            evidence_in_top5_count += 1

        chunks = [mem_texts[j] for j in top_idx]
        fs_ans = answer(q, "\n".join(f"- {c}" for c in chunks))

        st_ctx = "\n".join(f"- {mem_ids[j]}: {mem_texts[j][:200]}" for j in top_idx)
        st_ans = answer(q, st_ctx)

        traj_ctx = build_trajectory(evidence_ids, memory)
        traj_ans = answer(q, traj_ctx)

        fs_ok, _ = judge(q, fs_ans or "", gt)
        st_ok, _ = judge(q, st_ans or "", gt)
        traj_ok, _ = judge(q, traj_ans or "", gt)

        if fs_ok:
            fs_correct += 1
        if st_ok:
            st_correct += 1
        if traj_ok:
            traj_correct += 1

        details.append({
            "qa_index": i,
            "question": q,
            "ground_truth": gt,
            "fs_answer": fs_ans or "",
            "fs_correct": fs_ok,
            "state_answer": st_ans or "",
            "state_correct": st_ok,
            "traj_answer": traj_ans or "",
            "traj_correct": traj_ok,
            "evidence_ids": evidence_ids,
            "evidence_in_top5": evidence_in_top5,
        })

        # Save checkpoint every 50 QA
        if (i + 1) % 50 == 0:
            save_checkpoint(i + 1, N, details, fs_correct, st_correct, traj_correct,
                           evidence_in_top5_count, start_time)

        # Per-QA live progress
        elapsed = (datetime.datetime.utcnow() - start_time).total_seconds()
        rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
        eta = (N - i - 1) / rate if rate > 0 else 0
        bar_len = 30
        filled = int(bar_len * (i + 1 - start_idx) / (N - start_idx))
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{i+1:4d}/{N}] {bar} "
            f"FS={fs_correct/(i+1):.3f} "
            f"St={st_correct/(i+1):.3f} "
            f"Tj={traj_correct/(i+1):.3f} "
            f"Top5={evidence_in_top5_count/(i+1):.3f} "
            f"| {elapsed//60:.0f}m ETA {eta//60:.0f}m  ",
            end="", flush=True
        )

        # Newline with summary at save boundaries
        if (i + 1) % 50 == 0 or (i + 1) == N:
            print()

        time.sleep(0.12)

    # Final newline after progress bar
    print()

    fs_acc = fs_correct / N
    st_acc = st_correct / N
    traj_acc = traj_correct / N
    top5_rate = evidence_in_top5_count / N

    results = {
        "metadata": {
            "experiment_phase": "Full 1013 QA (§6.3 / Appendix C)",
            "script": "run_full1013.py",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "llm_model": MODEL,
            "llm_temperature": 0.0,
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": K,
            "n_qa": N,
            "qa_source": "atm-bench.json (full 1013 QA)",
            "evidence_field_email": "detail[:1000]",
            "evidence_field_media": "caption[:1000]",
            "trajectory_format": "chronologically sorted evidence with step indices",
            "system_prompt_snippet": ANSWER_SYSTEM_PROMPT[:200],
            "total_elapsed_seconds": (datetime.datetime.utcnow() - start_time).total_seconds(),
        },
        "n_qa": N,
        "fs_accuracy": round(fs_acc, 4),
        "state_accuracy": round(st_acc, 4),
        "trajectory_accuracy": round(traj_acc, 4),
        "evidence_in_top5_rate": round(top5_rate, 4),
        "usage": get_usage(),
        "details": details,
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS — Full 1013 QA (§6.3 / Appendix C)")
    print(f"{'='*60}")
    print(f"FS accuracy:              {fs_acc:.4f}")
    print(f"State accuracy:           {st_acc:.4f}")
    print(f"Trajectory accuracy:      {traj_acc:.4f}")
    print(f"Evidence in top-5 rate:   {top5_rate:.4f}")
    print(f"Usage: {results['usage']}")
    print(f"Elapsed: {results['metadata']['total_elapsed_seconds']:.0f}s")
    print(f"Results saved to: {RESULT_PATH}")

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
